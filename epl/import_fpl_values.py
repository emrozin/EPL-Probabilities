"""Build each Premier League team's squad value for every week, from Fantasy Premier League prices.

Usage:
    python -m epl.import_fpl_values

Source: the FPL Historical Dataset by Vaastav Anand (https://github.com/vaastav/Fantasy-Premier-League),
MIT license. It has every player's FPL price for every gameweek since 2016/17, and is kept up
to date during the season (with a lag of a few weeks).

A squad's value is the total FPL price of its TOP_PLAYERS most expensive players. Only
information available on each date is used: a gameweek's prices count from its first kickoff,
except gameweek 1, whose prices are published in the summer and count from July 1.

Each player's team comes from the matches themselves: within a match, the home players'
opponent is the away team and vice versa. FPL teams are linked to our team names by matching
matches on date and score.

Two versions are saved:
  fpl_squad_values            prices as they change during the season (with each gameweek)
  fpl_preseason_squad_values  each season's starting (gameweek 1) prices, used all season.
                              In-season price changes mostly follow fantasy managers' reactions
                              to recent form, which the model already sees in results.
"""

import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from epl.import_epl import LAST_SEASON
from epl.model import load_games

DB_PATH = Path("data/sports.db")
CACHE_DIR = Path("data/fpl")
BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
FIRST_SEASON = 2016
TOP_PLAYERS = 18
STALE_DAYS = 120            # don't use an in-season value more than about four months old
PRESEASON_STALE_DAYS = 364  # starting prices stand for the whole season, until the next one's


def season_name(season: int) -> str:
    """2016 -> '2016-17'"""
    return f"{season}-{(season + 1) % 100:02d}"


def load_season(season: int) -> pd.DataFrame:
    """One season's gameweek-by-gameweek player data, cached once the season is over."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{season_name(season)}_merged_gw.csv"
    if not path.exists() or season == LAST_SEASON:
        try:
            response = requests.get(f"{BASE_URL}/{season_name(season)}/gws/merged_gw.csv", timeout=60)
            response.raise_for_status()
            path.write_bytes(response.content)
        except requests.RequestException as err:
            if not path.exists():
                raise
            print(f"  Couldn't download {season_name(season)} ({err}); using the copy saved earlier.")
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df["season"] = season
    return df


def player_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per player per match, with the FPL ids of both teams and the player's own team."""
    home = df["was_home"].astype(str).str.lower() == "true"
    # Within a match, away players' opponent is the home team, and home players' opponent the away team.
    home_team = df[~home].groupby("fixture")["opponent_team"].first().rename("home_id")
    away_team = df[home].groupby("fixture")["opponent_team"].first().rename("away_id")
    rows = df.join(home_team, on="fixture").join(away_team, on="fixture")
    rows = rows.dropna(subset=["home_id", "away_id"]).copy()
    rows["team_id"] = np.where(home[rows.index], rows["home_id"], rows["away_id"]).astype(int)
    kickoff = pd.to_datetime(rows["kickoff_time"], utc=True)
    rows["kickoff"] = kickoff
    rows["date"] = kickoff.dt.tz_convert("Europe/London").dt.strftime("%Y-%m-%d")  # our dates are UK dates
    rows["price"] = rows["value"] / 10   # FPL prices are stored in tenths of a million
    return rows[["season", "GW", "fixture", "element", "price", "team_id", "home_id", "away_id",
                 "date", "kickoff", "team_h_score", "team_a_score"]]


def team_mapping(rows: pd.DataFrame, ours: pd.DataFrame) -> dict[tuple[int, int], str]:
    """(season, FPL team id) -> our team name, by matching matches on date and score."""
    fixtures = rows.drop_duplicates(["season", "fixture"])
    fixtures = fixtures.dropna(subset=["team_h_score", "team_a_score"])
    matched = fixtures.merge(ours, left_on=["date", "team_h_score", "team_a_score"],
                             right_on=["date", "home_score", "away_score"])
    votes: dict[tuple[int, int], Counter] = {}
    for fpl_col, our_col in (("home_id", "home"), ("away_id", "away")):
        for season, fpl_id, team in zip(matched["season"], matched[fpl_col], matched[our_col]):
            votes.setdefault((int(season), int(fpl_id)), Counter())[team] += 1
    return {key: counts.most_common(1)[0][0] for key, counts in votes.items()}


def gameweek_values(rows: pd.DataFrame, mapping: dict, top: int = TOP_PLAYERS) -> pd.DataFrame:
    """Each team's squad value per gameweek, and the date from which that value was known."""
    rows = rows.copy()
    rows["team"] = [mapping.get((s, t)) for s, t in zip(rows["season"], rows["team_id"])]
    rows = rows.dropna(subset=["team"])
    # A player with two matches in a gameweek appears twice; count him once.
    players = rows.drop_duplicates(["season", "GW", "element"])
    players = players.sort_values(["season", "GW", "team", "price"], ascending=[True, True, True, False])
    best = players.groupby(["season", "GW", "team"]).head(top)
    values = best.groupby(["season", "GW", "team"]).agg(value=("price", "sum"), players=("element", "size")).reset_index()

    first_kickoff = rows.groupby(["season", "GW"])["kickoff"].min().dt.tz_localize(None).dt.normalize()
    values["known_from"] = [pd.Timestamp(f"{s}-07-01") if gw == 1 else first_kickoff[(s, gw)]
                            for s, gw in zip(values["season"], values["GW"])]
    return values


def weekly_values(values: pd.DataFrame, weeks: pd.DatetimeIndex, stale_days: int = STALE_DAYS) -> pd.DataFrame:
    """For each Monday and team, the most recent squad value known by then (at most stale_days old)."""
    grid = pd.MultiIndex.from_product([weeks, values["team"].unique()], names=["week", "team"]).to_frame(index=False)
    known = values.sort_values("known_from")[["team", "known_from", "value", "players"]]
    out = pd.merge_asof(grid.sort_values("week"), known, left_on="week", right_on="known_from",
                        by="team", direction="backward")
    out = out.dropna(subset=["value"])
    return out[(out["week"] - out["known_from"]).dt.days <= stale_days]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    ours = load_games(conn)[["date", "home", "away", "home_score", "away_score"]]

    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        try:
            frames.append(player_rows(load_season(season)))
            print(f"{season_name(season)}: {frames[-1]['GW'].max()} gameweeks", flush=True)
        except requests.RequestException as err:
            print(f"{season_name(season)}: skipped ({err})")
    rows = pd.concat(frames, ignore_index=True)

    mapping = team_mapping(rows, ours)
    unlinked = sorted({(s, t) for s, t in zip(rows["season"], rows["team_id"])} - set(mapping))
    print(f"\nLinked {len(mapping)} season-teams to your team names"
          + (f"; couldn't link {len(unlinked)}: {unlinked}" if unlinked else "."))

    gw_values = gameweek_values(rows, mapping)
    weeks = pd.date_range(f"{FIRST_SEASON}-07-04", pd.Timestamp.today(), freq="W-MON")
    versions = {
        "fpl_squad_values": weekly_values(gw_values, weeks),
        "fpl_preseason_squad_values": weekly_values(gw_values[gw_values["GW"] == 1], weeks, PRESEASON_STALE_DAYS),
    }
    with conn:
        for table, weekly in versions.items():
            conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
                                league_code TEXT NOT NULL,
                                team        TEXT NOT NULL,
                                week        TEXT NOT NULL,   -- a Monday, 'YYYY-MM-DD'
                                value_eur   REAL NOT NULL,   -- total FPL price (in £m) of the TOP_PLAYERS most expensive players
                                players     INTEGER NOT NULL,
                                PRIMARY KEY (league_code, team, week))""")
            conn.execute(f"DELETE FROM {table} WHERE league_code = 'EPL'")
            conn.executemany(f"INSERT INTO {table} VALUES ('EPL', ?, ?, ?, ?)",
                             [(r.team, r.week.strftime("%Y-%m-%d"), float(r.value), int(r.players))
                              for r in weekly.itertuples()])
            print(f"Saved {len(weekly):,} weekly squad values to {table}.")
    print()

    for season in (2016, 2020, 2024):
        gw1 = gw_values[(gw_values["season"] == season) & (gw_values["GW"] == 1)].sort_values("value", ascending=False)
        if gw1.empty:
            continue
        fmt = lambda part: ", ".join(f"{r.team} £{r.value:.0f}m" for r in part.itertuples())
        print(f"{season_name(season)} gameweek 1\n  most expensive:  {fmt(gw1.head(4))}\n  least expensive: {fmt(gw1.tail(3))}")
    conn.close()


if __name__ == "__main__":
    main()
