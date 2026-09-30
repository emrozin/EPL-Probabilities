"""Import Understat expected goals (xG) for EPL matches, 2014/15 onward.

Usage (run import_epl.py first, since this links xG to games already in the database):
    python import_understat.py

Understat has no official API. Its league pages load their data from an internal
JSON endpoint, getLeagueData/<league>/<season>, whose `dates` list holds every
fixture with each team's xG. This script requests that endpoint directly.

Extracted data is cached in data/raw/understat/. Completed seasons are downloaded
once; the current season is refreshed on every run. Safe to re-run.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import requests

DB_PATH = Path("data/sports.db")
SCHEMA_PATH = Path("schema.sql")
ALIASES_PATH = Path("team_aliases.sql")
CACHE_DIR = Path("data/raw/understat")

SOURCE = "understat"
LEAGUE_CODE = "EPL"          # our league code
UNDERSTAT_LEAGUE = "EPL"     # the league's name in Understat URLs
FIRST_SEASON = 2014          # Understat's first season (2014/15)
LAST_SEASON = 2026           # current season

HEADERS = {"User-Agent": "Mozilla/5.0 (personal sports analytics project)"}
SECONDS_BETWEEN_REQUESTS = 3  # be polite to the site


REQUIRED_MATCH_KEYS = {"id", "isResult", "h", "a", "goals", "xG", "datetime"}


def fetch_season(season: int) -> list[dict]:
    """Return the list of fixtures for one season, using the cache if possible."""
    cache_file = CACHE_DIR / f"{UNDERSTAT_LEAGUE}_{season}.json"
    if cache_file.exists() and season != LAST_SEASON:
        return json.loads(cache_file.read_text())

    url = f"https://understat.com/getLeagueData/{UNDERSTAT_LEAGUE}/{season}"
    # This header marks the request as the page's own data request (AJAX);
    # without it the endpoint doesn't return the JSON.
    headers = {**HEADERS, "X-Requested-With": "XMLHttpRequest"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as err:
        if not cache_file.exists():
            raise
        print(f"  Couldn't download the latest data ({err}); using the copy saved earlier.")
        return json.loads(cache_file.read_text())

    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("Understat didn't return JSON. The endpoint may have changed.")
    matches = payload.get("dates") if isinstance(payload, dict) else None
    if not isinstance(matches, list):
        raise RuntimeError("Understat's response has no 'dates' list. The format may have changed.")
    if matches and not REQUIRED_MATCH_KEYS <= set(matches[0]):
        raise RuntimeError(f"Understat's match format changed. Fields found: {sorted(matches[0])}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(matches, indent=1))
    time.sleep(SECONDS_BETWEEN_REQUESTS)
    return matches


def load_team_lookup(conn: sqlite3.Connection, league_id: int) -> dict[str, int]:
    """Map every name we can recognize (our names + aliases) to a team id."""
    lookup = {
        name: team_id
        for team_id, name in conn.execute("SELECT id, name FROM teams WHERE league_id = ?", (league_id,))
    }
    for source_name, team_id in conn.execute(
        "SELECT source_name, team_id FROM team_aliases WHERE source = ? AND league_id = ?",
        (SOURCE, league_id),
    ):
        lookup[source_name] = team_id
    return lookup


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    conn.executescript(ALIASES_PATH.read_text())

    row = conn.execute("SELECT id FROM leagues WHERE code = ?", (LEAGUE_CODE,)).fetchone()
    if row is None:
        raise SystemExit("No EPL data found. Run import_epl.py first.")
    league_id = row[0]
    lookup = load_team_lookup(conn, league_id)

    unknown_teams: dict[str, set[int]] = {}  # Understat name -> seasons it appeared in
    unlinked: list[str] = []                 # matches we couldn't pair with one of our games
    score_mismatches: list[str] = []         # linked, but the final score disagrees
    skipped: list[str] = []                  # seasons that couldn't be downloaded

    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        label = f"{season}/{(season + 1) % 100:02d}"
        try:
            matches = fetch_season(season)
        except (requests.RequestException, RuntimeError) as err:
            print(f"{label}: skipped ({err})")
            skipped.append(label)
            continue

        played = [m for m in matches if m.get("isResult")]
        linked = 0
        for m in played:
            home_name, away_name = m["h"]["title"], m["a"]["title"]
            home_id, away_id = lookup.get(home_name), lookup.get(away_name)
            for name, team_id in ((home_name, home_id), (away_name, away_id)):
                if team_id is None:
                    unknown_teams.setdefault(name, set()).add(season)
            if home_id is None or away_id is None:
                continue

            # In a league season each home/away pairing happens exactly once,
            # so season + teams identifies the game without relying on dates.
            games = conn.execute(
                """
                SELECT id, home_score, away_score FROM games
                WHERE league_id = ? AND season = ? AND home_team_id = ? AND away_team_id = ?
                """,
                (league_id, season, home_id, away_id),
            ).fetchall()
            if len(games) != 1:
                unlinked.append(f"{label} {home_name} v {away_name} ({m['datetime'][:10]}): "
                                f"{len(games)} matching games in database")
                continue

            game_id, home_score, away_score = games[0]
            if (int(m["goals"]["h"]), int(m["goals"]["a"])) != (home_score, away_score):
                score_mismatches.append(f"{label} {home_name} v {away_name}: Understat "
                                        f"{m['goals']['h']}-{m['goals']['a']}, ours {home_score}-{away_score}")

            conn.execute(
                """
                INSERT INTO game_xg (game_id, source, home_xg, away_xg, source_match_id)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (game_id, source) DO UPDATE SET
                    home_xg = excluded.home_xg,
                    away_xg = excluded.away_xg,
                    source_match_id = excluded.source_match_id
                """,
                (game_id, SOURCE, float(m["xG"]["h"]), float(m["xG"]["a"]), str(m["id"])),
            )
            linked += 1

        conn.commit()
        print(f"{label}: {linked} of {len(played)} played matches linked")

    conn.close()

    if unknown_teams:
        print("\nUnderstat team names with no match (add them to team_aliases.sql):")
        for name, seasons in sorted(unknown_teams.items()):
            print(f"  {name}  (seasons {min(seasons)}-{max(seasons)})")
    if unlinked:
        print(f"\nMatches not linked to a game ({len(unlinked)}):")
        for line in unlinked[:20]:
            print("  " + line)
        if len(unlinked) > 20:
            print(f"  ... and {len(unlinked) - 20} more")
    if score_mismatches:
        print(f"\nLinked matches where the scores disagree ({len(score_mismatches)}):")
        for line in score_mismatches[:20]:
            print("  " + line)
    if skipped:
        print(f"\n{len(skipped)} season(s) skipped, so nothing was imported for them: {', '.join(skipped)}")
        if len(skipped) == LAST_SEASON - FIRST_SEASON + 1:
            # Every season failed: Understat is unreachable or has changed. Stop the update rather
            # than let the model quietly fall back to goals-only for every match.
            sys.exit(1)
    elif not (unknown_teams or unlinked or score_mismatches):
        print("\nAll played matches linked, and every score agrees.")


if __name__ == "__main__":
    main()
