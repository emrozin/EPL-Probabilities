"""Import upcoming EPL fixtures and pre-match odds from football-data.co.uk.

Usage:
    python import_fixtures.py

fixtures.csv lists the next round of matches for many leagues, with odds collected
Friday afternoons for weekend games and Tuesday afternoons for midweek games.
Fixtures are stored as games with status 'scheduled'; when import_epl.py later
imports the result, the same game row is updated to 'final'. Safe to re-run.
"""

import io
import sqlite3

import pandas as pd
import requests

from import_epl import DB_PATH, SCHEMA_PATH, get_league_id, insert_odds

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
DIVISION = "E0"  # Premier League's code in football-data.co.uk files


def season_for(match_date: pd.Timestamp) -> int:
    """Seasons start in August: a match in March 2027 belongs to season 2026 (2026/27)."""
    return match_date.year if match_date.month >= 7 else match_date.year - 1


def main() -> None:
    response = requests.get(FIXTURES_URL, timeout=30)
    response.raise_for_status()
    text = response.content.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), on_bad_lines="warn")

    df = df[df["Div"] == DIVISION].dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    if df.empty:
        print("No Premier League fixtures in the file right now (for example, during an international break).")
        return
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    league_id = get_league_id(conn)
    team_ids = {name: team_id for team_id, name in
                conn.execute("SELECT id, name FROM teams WHERE league_id = ?", (league_id,))}

    new_games = new_odds = 0
    for _, row in df.iterrows():
        home, away = row["HomeTeam"].strip(), row["AwayTeam"].strip()
        unknown = [team for team in (home, away) if team not in team_ids]
        if unknown:
            # Don't create teams from this file: a typo would silently become a new team.
            print(f"  Skipped {home} v {away}: unknown team name(s) {unknown}")
            continue

        match_date = row["Date"].strftime("%Y-%m-%d")
        cursor = conn.execute(
            """
            INSERT INTO games (league_id, season, date, home_team_id, away_team_id, status)
            VALUES (?, ?, ?, ?, ?, 'scheduled')
            ON CONFLICT (league_id, season, date, home_team_id, away_team_id) DO NOTHING
            """,
            (league_id, season_for(row["Date"]), match_date, team_ids[home], team_ids[away]),
        )
        new_games += cursor.rowcount

        game_id = conn.execute(
            """
            SELECT id FROM games
            WHERE league_id = ? AND date = ? AND home_team_id = ? AND away_team_id = ?
            """,
            (league_id, match_date, team_ids[home], team_ids[away]),
        ).fetchone()[0]
        added = insert_odds(conn, game_id, row)
        new_odds += added
        print(f"  {match_date}  {home} v {away}  ({added} new odds rows)")

    conn.commit()
    conn.close()
    print(f"{len(df)} Premier League fixtures in the file: {new_games} new games, {new_odds} new odds rows")


if __name__ == "__main__":
    main()
