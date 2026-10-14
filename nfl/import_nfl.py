"""Import NFL games, results and betting lines from nflverse into data/sports.db.

Usage:
    python -m nfl.import_nfl

Source: nflverse's games file, one row per game since 1999 (regular season and playoffs),
with scores, the spread, total and moneylines, and details such as rest days, starting
quarterbacks and weather. It includes the whole current season's schedule; games not yet
played are stored with status 'scheduled'. Safe to re-run: existing games are updated.

The spread and total (every season) are kept in nfl_game_info, for every game. Prices
(from 2006) go in the odds table, for finished games only. nflverse updates a game's lines
until kickoff, so for a finished game they are the last lines recorded before it was played:
treated here as closing lines. Its documentation doesn't name the sportsbook, so the
bookmaker is recorded as 'nflverse'.
"""

import io
import sqlite3

import pandas as pd
import requests

from db import DB_PATH, RAW_DIR, SCHEMA_PATH, as_int, get_team_id, insert_price

URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
RAW_PATH = RAW_DIR / "nfl_games.csv"
LEAGUE = ("NFL", "National Football League", "american_football")
BOOKMAKER = "nflverse"

# Teams that moved: each franchise keeps one name, so its history carries over.
FRANCHISES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

INFO_COLUMNS = {  # nfl_game_info column -> nflverse column
    "game_type": "game_type", "week": "week", "overtime": "overtime",
    "home_rest": "home_rest", "away_rest": "away_rest", "div_game": "div_game",
    "roof": "roof", "surface": "surface", "temp": "temp", "wind": "wind",
    "home_qb": "home_qb_name", "away_qb": "away_qb_name",
    "home_coach": "home_coach", "away_coach": "away_coach",
    "referee": "referee", "stadium": "stadium",
    "spread_line": "spread_line", "total_line": "total_line",
}
INTEGER_INFO = {"week", "overtime", "home_rest", "away_rest", "div_game", "temp", "wind"}
FLOAT_INFO = {"spread_line", "total_line"}
REQUIRED = ["game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
            "home_score", "away_score", "location", "spread_line", "total_line"]


def download() -> pd.DataFrame:
    """Download the games file (it changes every week), keeping a copy in data/raw/."""
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH.write_bytes(response.content)
    return read_games(response.content.decode("utf-8-sig"))


def read_games(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        # Fail loudly if nflverse changes its format, rather than importing nothing.
        raise SystemExit(f"The nflverse games file has changed: missing columns {missing}")
    return df


def american_to_decimal(odds):
    """-150 -> 1.667 (bet 150 to win 100); +130 -> 2.30 (bet 100 to win 130). None if missing."""
    if odds is None or pd.isna(odds) or odds == 0:
        return None
    odds = float(odds)
    return 1 + (100 / -odds if odds < 0 else odds / 100)


def get_league_id(conn) -> int:
    conn.execute("INSERT OR IGNORE INTO leagues (code, name, sport) VALUES (?, ?, ?)", LEAGUE)
    return conn.execute("SELECT id FROM leagues WHERE code = ?", (LEAGUE[0],)).fetchone()[0]


def text(value):
    return None if value is None or pd.isna(value) else str(value)


def upsert_game(conn, league_id, row, home_id, away_id) -> tuple[int, bool]:
    """Insert or update one game, matched by nflverse's game id. Returns (game id, whether it's new).

    The same home/away pairing can happen twice in a season (a playoff rematch), so season and
    teams alone don't identify an NFL game the way they do in a soccer league.
    """
    played = pd.notna(row["home_score"]) and pd.notna(row["away_score"])
    values = (row["gameday"], text(row.get("gametime")),
              as_int(row["home_score"]) if played else None,
              as_int(row["away_score"]) if played else None,
              "final" if played else "scheduled")
    existing = conn.execute("SELECT game_id FROM nfl_game_info WHERE source_game_id = ?",
                            (row["game_id"],)).fetchone()
    if existing:
        conn.execute("UPDATE games SET date = ?, kickoff = ?, home_score = ?, away_score = ?, status = ? "
                     "WHERE id = ?", (*values, existing[0]))
        return existing[0], False
    game_id = conn.execute(
        """INSERT INTO games (league_id, season, date, kickoff, home_score, away_score, status,
                              home_team_id, away_team_id)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (league_id, int(row["season"]), *values, home_id, away_id)).lastrowid
    return game_id, True


def upsert_info(conn, game_id, row) -> None:
    info = {"source_game_id": row["game_id"], "neutral_site": int(row["location"] == "Neutral")}
    for column, source in INFO_COLUMNS.items():
        value = row.get(source)
        if column in INTEGER_INFO:
            info[column] = as_int(value)
        elif column in FLOAT_INFO:
            info[column] = None if pd.isna(value) else float(value)
        else:
            info[column] = text(value)
    columns = ", ".join(info)
    conn.execute(f"INSERT OR REPLACE INTO nfl_game_info (game_id, {columns}) "
                 f"VALUES (?, {', '.join('?' for _ in info)})", (game_id, *info.values()))


def insert_lines(conn, game_id, row) -> int:
    """The spread, total and moneyline for a finished game. Returns how many new odds rows were added.

    Stored like the soccer Asian handicap: the line is the HOME team's handicap on both rows,
    so nflverse's spread_line of +3.5 (home favored by 3.5) is stored as -3.5.
    """
    added = 0
    if pd.notna(row["spread_line"]):
        line = -float(row["spread_line"])
        added += insert_price(conn, game_id, BOOKMAKER, "spread", "home",
                              american_to_decimal(row.get("home_spread_odds")), line, 1)
        added += insert_price(conn, game_id, BOOKMAKER, "spread", "away",
                              american_to_decimal(row.get("away_spread_odds")), line, 1)
    if pd.notna(row["total_line"]):
        line = float(row["total_line"])
        added += insert_price(conn, game_id, BOOKMAKER, "total", "over",
                              american_to_decimal(row.get("over_odds")), line, 1)
        added += insert_price(conn, game_id, BOOKMAKER, "total", "under",
                              american_to_decimal(row.get("under_odds")), line, 1)
    for outcome in ("home", "away"):
        added += insert_price(conn, game_id, BOOKMAKER, "moneyline", outcome,
                              american_to_decimal(row.get(f"{outcome}_moneyline")), None, 1)
    return added


def import_games(conn, df: pd.DataFrame) -> dict:
    conn.executescript(SCHEMA_PATH.read_text())
    league_id = get_league_id(conn)
    cache: dict = {}
    counts = {"new": 0, "updated": 0, "final": 0, "scheduled": 0, "odds": 0}
    for _, row in df.iterrows():
        home = FRANCHISES.get(row["home_team"], row["home_team"])
        away = FRANCHISES.get(row["away_team"], row["away_team"])
        game_id, is_new = upsert_game(conn, league_id, row,
                                      get_team_id(conn, league_id, home, cache),
                                      get_team_id(conn, league_id, away, cache))
        counts["new" if is_new else "updated"] += 1
        upsert_info(conn, game_id, row)
        conn.execute("INSERT OR REPLACE INTO raw_source_rows (game_id, source, data) VALUES (?, ?, ?)",
                     (game_id, "nflverse games.csv", row.dropna().to_json()))
        if pd.notna(row["home_score"]):
            counts["final"] += 1
            counts["odds"] += insert_lines(conn, game_id, row)
        else:
            counts["scheduled"] += 1
    conn.commit()
    return counts


def main() -> None:
    df = download()
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    counts = import_games(conn, df)
    teams = conn.execute("SELECT COUNT(*) FROM teams t JOIN leagues l ON l.id = t.league_id "
                         "WHERE l.code = 'NFL'").fetchone()[0]
    conn.close()
    print(f"{len(df)} NFL games in the file, {df['season'].min()} to {df['season'].max()}: "
          f"{counts['final']} played, {counts['scheduled']} still to play.")
    print(f"{counts['new']} new games, {counts['updated']} updated, {counts['odds']} new odds rows, {teams} teams.")


if __name__ == "__main__":
    main()
