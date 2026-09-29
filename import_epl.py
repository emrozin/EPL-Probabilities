"""Import every EPL season from football-data.co.uk into SQLite.

Usage:
    python import_epl.py

Imports results, match stats, and all odds the files contain (1X2, over/under 2.5,
Asian handicap), and keeps every original row as JSON so nothing is lost.

Downloaded CSVs are cached in data/raw/. Completed seasons are only downloaded once;
the current season is re-downloaded on every run to pick up new results.
Safe to re-run: games are upserted and existing odds rows are skipped.
"""

import io
import sqlite3
from pathlib import Path

import pandas as pd
import requests

DB_PATH = Path("data/sports.db")
RAW_DIR = Path("data/raw")
SCHEMA_PATH = Path("schema.sql")

# Seasons are stored by start year: 2024 means 2024/25.
FIRST_SEASON = 1993   # the first season football-data.co.uk has for the EPL
LAST_SEASON = 2026    # current season; only matches played so far are in the file

# ---------------------------------------------------------------------------
# Odds column naming in the CSVs:
#   1X2:            <prefix>H / D / A          e.g. PSH, B365D
#   over/under 2.5: <prefix>>2.5 / <2.5        e.g. B365>2.5, P<2.5
#   Asian handicap: <prefix>AHH / AHA, line in AHh (older files: BbAHh)
#   Closing odds:   a "C" right after the prefix: PSCH, B365C>2.5, PCAHH, line AHCh
# Older files use BbAv / BbMx for the market average / maximum.
# Columns a season doesn't have are simply skipped.
# ---------------------------------------------------------------------------
BOOKMAKERS_1X2 = {
    "Pinnacle": ["PS"],
    "Bet365": ["B365"],
    "Bet&Win": ["BW"],
    "Interwetten": ["IW"],
    "William Hill": ["WH"],
    "VC Bet": ["VC"],
    "Ladbrokes": ["LB"],
    "Stan James": ["SJ"],
    "Gamebookers": ["GB"],
    "Blue Square": ["BS"],
    "Sportingbet": ["SB"],
    "Sporting Odds": ["SO"],
    "1xBet": ["1XB"],
    "Betfred": ["BFD"],
    "BetVictor": ["BV"],
    "Paddy Power": ["PP"],
    "Sky Bet": ["SKB"],
    "Betfair Exchange": ["BFE"],
    "Betfair": ["BF"],
    "BetMGM": ["BMGM"],
    "Coral": ["CL"],
    "Stanleybet": ["SY"],
    "Market max": ["Max", "BbMx"],
    "Market average": ["Avg", "BbAv"],
}
BOOKMAKERS_TOTALS_AH = {
    "Pinnacle": ["P"],
    "Bet365": ["B365"],
    "Betfair Exchange": ["BFE"],
    "Market max": ["Max", "BbMx"],
    "Market average": ["Avg", "BbAv"],
}

# CSV column -> soccer_match_stats column
STATS_FLOAT_COLUMNS = {"HxG": "home_xg", "AxG": "away_xg"}
STATS_COLUMNS = {
    "HTHG": "ht_home_score", "HTAG": "ht_away_score",
    "HS": "home_shots", "AS": "away_shots",
    "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HF": "home_fouls", "AF": "away_fouls",
    "HC": "home_corners", "AC": "away_corners",
    "HY": "home_yellows", "AY": "away_yellows",
    "HR": "home_reds", "AR": "away_reds",
}


def season_code(season: int) -> str:
    """2024 -> '2425'"""
    return f"{season % 100:02d}{(season + 1) % 100:02d}"


def load_season(season: int) -> pd.DataFrame:
    """Return one season's CSV as a DataFrame (think: in-memory table), using the cache if possible."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = RAW_DIR / f"E0_{season_code(season)}.csv"

    if not cache_file.exists() or season == LAST_SEASON:
        url = f"https://www.football-data.co.uk/mmz4281/{season_code(season)}/E0.csv"
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # throws an exception on a 404 etc.
        cache_file.write_bytes(response.content)

    text = cache_file.read_bytes().decode("utf-8-sig", errors="replace")

    # Some seasons have rows with more fields than the header line (columns were added
    # mid-season without updating the header). Give those extra fields placeholder
    # names so no row gets skipped; any non-empty extras are kept in the raw JSON.
    lines = [line for line in text.splitlines() if line.strip()]
    header = [name.strip() or f"unnamed_{i}" for i, name in enumerate(lines[0].split(","))]
    max_fields = max(line.count(",") + 1 for line in lines)
    names = header + [f"extra_{i}" for i in range(max_fields - len(header))]
    df = pd.read_csv(io.StringIO("\n".join(lines[1:])), header=None, names=names)

    # Drop placeholder columns that are completely empty.
    placeholder = [c for c in df.columns if c.startswith(("unnamed_", "extra_"))]
    df = df.drop(columns=[c for c in placeholder if df[c].isna().all()])

    # Drop blank rows, which some files have at the end.
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"])

    # Dates are dd/mm/yy or dd/mm/yyyy depending on the season; store ISO format.
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed").dt.strftime("%Y-%m-%d")
    return df


def first_value(row: pd.Series, columns: list[str]):
    """Return the first column in `columns` that exists in this row and isn't empty."""
    for col in columns:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def known_columns() -> set[str]:
    """Every CSV column this importer maps into a table (everything else stays in the raw JSON)."""
    known = {"Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HTR",
             "Referee", "AHh", "BbAHh", "AHCh"}
    known |= set(STATS_COLUMNS) | set(STATS_FLOAT_COLUMNS)
    for c in ("", "C"):
        for prefixes in BOOKMAKERS_1X2.values():
            known |= {p + c + letter for p in prefixes for letter in "HDA"}
        for prefixes in BOOKMAKERS_TOTALS_AH.values():
            known |= {p + c + s for p in prefixes for s in (">2.5", "<2.5", "AHH", "AHA")}
    return known


def as_int(value):
    return None if value is None or pd.isna(value) else int(value)


def get_league_id(conn: sqlite3.Connection) -> int:
    conn.execute(
        "INSERT OR IGNORE INTO leagues (code, name, sport) VALUES (?, ?, ?)",
        ("EPL", "Premier League", "soccer"),
    )
    return conn.execute("SELECT id FROM leagues WHERE code = ?", ("EPL",)).fetchone()[0]


def get_team_id(conn: sqlite3.Connection, league_id: int, name: str, cache: dict) -> int:
    """Return the team's id, inserting it first if needed. `cache` avoids repeat lookups."""
    name = name.strip()
    if name not in cache:
        conn.execute(
            "INSERT OR IGNORE INTO teams (league_id, name) VALUES (?, ?)",
            (league_id, name),
        )
        cache[name] = conn.execute(
            "SELECT id FROM teams WHERE league_id = ? AND name = ?",
            (league_id, name),
        ).fetchone()[0]
    return cache[name]


def upsert_game(conn, league_id, season, date, home_id, away_id, home_score, away_score) -> int:
    conn.execute(
        """
        INSERT INTO games (league_id, season, date, home_team_id, away_team_id,
                           home_score, away_score, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, 'final')
        ON CONFLICT (league_id, season, date, home_team_id, away_team_id)
        DO UPDATE SET home_score = excluded.home_score,
                      away_score = excluded.away_score,
                      status     = 'final'
        """,
        (league_id, season, date, home_id, away_id, home_score, away_score),
    )
    return conn.execute(
        """
        SELECT id FROM games
        WHERE league_id = ? AND season = ? AND date = ?
          AND home_team_id = ? AND away_team_id = ?
        """,
        (league_id, season, date, home_id, away_id),
    ).fetchone()[0]


def upsert_stats(conn: sqlite3.Connection, game_id: int, row: pd.Series) -> None:
    values = {db_col: as_int(first_value(row, [csv_col])) for csv_col, db_col in STATS_COLUMNS.items()}
    for csv_col, db_col in STATS_FLOAT_COLUMNS.items():
        value = first_value(row, [csv_col])
        values[db_col] = float(value) if value is not None else None
    referee = first_value(row, ["Referee"])
    values["referee"] = str(referee).strip() if referee is not None else None

    if all(v is None for v in values.values()):
        return  # early seasons have no stats at all

    columns = ", ".join(values)
    placeholders = ", ".join("?" for _ in values)
    conn.execute(
        f"INSERT OR REPLACE INTO soccer_match_stats (game_id, {columns}) VALUES (?, {placeholders})",
        (game_id, *values.values()),
    )


def save_raw_row(conn: sqlite3.Connection, game_id: int, season: int, row: pd.Series) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO raw_source_rows (game_id, source, data) VALUES (?, ?, ?)",
        (game_id, f"football-data.co.uk E0 {season}", row.dropna().to_json()),
    )


def insert_price(conn, game_id, bookmaker, market, outcome, price, line, is_closing) -> int:
    """Insert one price. Returns 1 if a new row was added, 0 otherwise."""
    if price is None or line is None and market != "1x2":
        return 0
    try:
        price = float(price)
    except (TypeError, ValueError):
        return 0  # the occasional junk value in old files
    if price <= 1.0:
        return 0  # not a valid decimal price
    cursor = conn.execute(
        """
        INSERT OR IGNORE INTO odds (game_id, bookmaker, market, outcome, price, line, is_closing)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (game_id, bookmaker, market, outcome, price, line, is_closing),
    )
    return cursor.rowcount


def insert_odds(conn: sqlite3.Connection, game_id: int, row: pd.Series) -> int:
    """Insert every price found in this row. Returns the number of new rows."""
    added = 0
    for is_closing, c in ((0, ""), (1, "C")):
        # 1X2
        for bookmaker, prefixes in BOOKMAKERS_1X2.items():
            for letter, outcome in (("H", "home"), ("D", "draw"), ("A", "away")):
                price = first_value(row, [p + c + letter for p in prefixes])
                added += insert_price(conn, game_id, bookmaker, "1x2", outcome, price, None, is_closing)

        # Over/under 2.5 goals, and Asian handicap
        ah_line = first_value(row, ["AHCh"] if c else ["AHh", "BbAHh"])
        ah_line = float(ah_line) if ah_line is not None else None
        for bookmaker, prefixes in BOOKMAKERS_TOTALS_AH.items():
            for symbol, outcome in ((">2.5", "over"), ("<2.5", "under")):
                price = first_value(row, [p + c + symbol for p in prefixes])
                added += insert_price(conn, game_id, bookmaker, "total", outcome, price, 2.5, is_closing)
            for suffix, outcome in (("AHH", "home"), ("AHA", "away")):
                price = first_value(row, [p + c + suffix for p in prefixes])
                added += insert_price(conn, game_id, bookmaker, "asian_handicap", outcome, price, ah_line, is_closing)
    return added


def main() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())

    league_id = get_league_id(conn)
    team_ids: dict[str, int] = {}
    known = known_columns()
    unmapped: dict[str, list[int]] = {}  # column -> seasons it appears in

    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        label = f"{season}/{(season + 1) % 100:02d}"
        try:
            df = load_season(season)
        except requests.RequestException as err:
            print(f"{label}: skipped ({err})")
            continue

        for col in df.columns:
            if col not in known and df[col].notna().any():
                unmapped.setdefault(col, []).append(season)

        new_odds = 0
        for _, row in df.iterrows():
            home_id = get_team_id(conn, league_id, row["HomeTeam"], team_ids)
            away_id = get_team_id(conn, league_id, row["AwayTeam"], team_ids)
            game_id = upsert_game(
                conn, league_id, season, row["Date"],
                home_id, away_id, int(row["FTHG"]), int(row["FTAG"]),
            )
            upsert_stats(conn, game_id, row)
            save_raw_row(conn, game_id, season, row)
            new_odds += insert_odds(conn, game_id, row)

        conn.commit()  # one transaction per season
        print(f"{label}: {len(df)} games, {new_odds} new odds rows")

    conn.close()

    if unmapped:
        print("\nColumns kept only in raw_source_rows (not mapped to a table):")
        for col, seasons in sorted(unmapped.items()):
            print(f"  {col:<10} {len(seasons):>2} season(s), {min(seasons)}-{max(seasons)}")


if __name__ == "__main__":  # Python's equivalent of Java's main()
    main()
