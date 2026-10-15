"""Import NFL weekly team and player stats from nflverse into data/sports.db.

Usage:
    python -m nfl.import_nfl_stats            # every season since 1999 (first run: ~60 downloads)

Source: nflverse's stats releases, one file per season for teams and one for players (passing,
rushing and receiving numbers, with expected points added). Past seasons are downloaded once and
cached in data/raw/nfl_stats/; the current season is downloaded again every run. Rows are linked
to games by nflverse's game id, so run python -m nfl.import_nfl first. Safe to re-run.
"""

import io
import sqlite3
from datetime import date

import pandas as pd
import requests

from db import DB_PATH, RAW_DIR, SCHEMA_PATH
from nfl.import_nfl import FRANCHISES

BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
FILES = {"team": "stats_team/stats_team_week_{season}.csv", "player": "stats_player/stats_player_week_{season}.csv"}
CACHE_DIR = RAW_DIR / "nfl_stats"
FIRST_SEASON = 1999

TEAM_COLUMNS = {"attempts": "pass_attempts", "sacks_suffered": "sacks", "carries": "carries",
                "passing_yards": "passing_yards", "rushing_yards": "rushing_yards",
                "passing_epa": "passing_epa", "rushing_epa": "rushing_epa"}
PLAYER_COLUMNS = {"completions": "completions", "attempts": "pass_attempts", "passing_yards": "passing_yards",
                  "passing_tds": "passing_tds", "passing_interceptions": "interceptions",
                  "carries": "carries", "rushing_yards": "rushing_yards", "rushing_tds": "rushing_tds",
                  "targets": "targets", "receptions": "receptions", "receiving_yards": "receiving_yards",
                  "receiving_tds": "receiving_tds"}
REQUIRED = {"team": ["game_id", "team", *TEAM_COLUMNS], "player": ["game_id", "team", "player_id",
                                                                    "player_display_name", "position", *PLAYER_COLUMNS]}


def current_season(today: date | None = None) -> int:
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def fetch(kind: str, season: int, refresh: bool) -> pd.DataFrame | None:
    """One season's file, from the cache unless it's missing or `refresh`. None if nflverse has no file."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{kind}_{season}.csv"
    if refresh or not path.exists():
        response = requests.get(f"{BASE_URL}/{FILES[kind].format(season=season)}", timeout=120)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        path.write_bytes(response.content)
    df = pd.read_csv(io.BytesIO(path.read_bytes()), low_memory=False)
    missing = [c for c in REQUIRED[kind] if c not in df.columns]
    if missing:
        raise SystemExit(f"The nflverse {kind} stats file for {season} has changed: missing columns {missing}")
    return df


def store(conn, kind: str, df: pd.DataFrame, game_ids: dict) -> int:
    """Write one season's rows, replacing any already stored for those games. Returns rows written."""
    columns = TEAM_COLUMNS if kind == "team" else PLAYER_COLUMNS
    keys = ["game_id", "team"] + (["player_id", "player_display_name", "position"] if kind == "player" else [])
    df = df[keys + list(columns)].dropna(subset=["game_id", "team"]).copy()
    df["team"] = df["team"].replace(FRANCHISES)
    df["game_id"] = df["game_id"].map(game_ids)
    df = df.dropna(subset=["game_id"]).rename(columns={**columns, "player_display_name": "name"})
    if kind == "player":
        df = df.dropna(subset=["player_id"])
        df["name"] = df["name"].fillna(df["player_id"])   # a few old rows have no name
        involved = df[["pass_attempts", "carries", "targets"]].fillna(0).sum(axis=1) > 0
        df = df[involved]                                   # offensive players only
    epa = {"passing_epa", "rushing_epa"}
    rows = [tuple(None if pd.isna(v) else (v if c in epa or isinstance(v, str) else int(v))
                  for c, v in zip(df.columns, r)) for r in df.itertuples(index=False)]
    table = f"nfl_{kind}_stats"
    conn.executemany(f"DELETE FROM {table} WHERE game_id = ?", [(int(g),) for g in set(df["game_id"])])
    conn.executemany(f"INSERT OR REPLACE INTO {table} ({', '.join(df.columns)}) VALUES "
                     f"({', '.join('?' for _ in df.columns)})", rows)
    return len(rows)


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    game_ids = dict(conn.execute("SELECT source_game_id, game_id FROM nfl_game_info"))
    if not game_ids:
        raise SystemExit("No NFL games found. Run python -m nfl.import_nfl first.")
    latest = current_season()
    totals = {"team": 0, "player": 0}
    for season in range(FIRST_SEASON, latest + 1):
        for kind in ("team", "player"):
            df = fetch(kind, season, refresh=season == latest)
            if df is not None:
                totals[kind] += store(conn, kind, df, game_ids)
        conn.commit()
    conn.close()
    print(f"Stored {totals['team']:,} team-game and {totals['player']:,} player-game stat rows, {FIRST_SEASON}-{latest}.")


if __name__ == "__main__":
    main()
