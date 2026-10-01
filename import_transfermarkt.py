"""Build each Premier League club's squad market value for every week, from Transfermarkt data.

Usage:
    python inspect_transfermarkt.py   # downloads the tables (once)
    python import_transfermarkt.py

Source: transfermarkt-datasets (CC0 license). Its updates paused in mid-July 2026, so squad
values stop at June 2026 and don't include this season's transfers.

Only information available on each date is used:
  - A player's club on a date is the club of his most recent transfer or appearance before it.
    (The valuations table's club columns describe a player's current club, not his club at the
    time, so they're deliberately not used.)
  - His value is his most recent valuation before that date, and he counts only if that
    valuation is recent: Transfermarkt keeps re-valuing active players, including those in
    leagues this dataset has no matches for (like the Championship), and stops for retired ones.
A squad's value is the total of its TOP_PLAYERS most valuable players (roughly a matchday
squad), so a big squad doesn't count as stronger just for having more players.

Transfermarkt clubs are linked to our team names by matching Premier League games (same date
and score). Results go to the squad_values table: one row per team per Monday.
"""

import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from inspect_transfermarkt import download
from model import load_games

DB_PATH = Path("data/sports.db")
LEAGUE_CODE = "EPL"
TM_COMPETITION = "GB1"
TOP_PLAYERS = 18
VALUATION_MAX_DAYS = 548   # a player counts only with a valuation from the last 18 months
FIRST_WEEK = "2013-07-01"  # appearances start in 2012, so from 2013 a year of history exists


def read(table: str) -> pd.DataFrame:
    return pd.read_csv(download(table), compression="gzip", low_memory=False)


def club_mapping(tm_games: pd.DataFrame, our_games: pd.DataFrame) -> dict[int, str]:
    """Transfermarkt club id -> our team name, by matching Premier League games on date and score."""
    tm = tm_games[tm_games["competition_id"] == TM_COMPETITION]
    matched = tm.merge(our_games, left_on=["date", "home_club_goals", "away_club_goals"],
                       right_on=["date", "home_score", "away_score"])
    votes: dict[int, Counter] = {}
    for tm_col, our_col in (("home_club_id", "home"), ("away_club_id", "away")):
        for club_id, team in zip(matched[tm_col], matched[our_col]):
            votes.setdefault(int(club_id), Counter())[team] += 1
    # Two games on the same day can have the same score, so take each club's most common match.
    return {club_id: counts.most_common(1)[0][0] for club_id, counts in votes.items()}


def weekly_dates(first: str, last: str) -> pd.DatetimeIndex:
    """Every Monday: matches are sometimes played in June or July (e.g. the 2020 Covid restart)."""
    return pd.date_range(first, last, freq="W-MON")


def squad_values(transfers: pd.DataFrame, appearances: pd.DataFrame, valuations: pd.DataFrame,
                 club_ids: set[int], weeks: pd.DatetimeIndex, top: int = TOP_PLAYERS) -> pd.DataFrame:
    """Columns: week, club_id, value_eur, players. One row per club per week."""
    events = pd.concat([
        pd.DataFrame({"player_id": transfers["player_id"], "event_date": pd.to_datetime(transfers["transfer_date"]),
                      "club_id": transfers["to_club_id"]}),
        pd.DataFrame({"player_id": appearances["player_id"], "event_date": pd.to_datetime(appearances["date"]),
                      "club_id": appearances["player_club_id"]}),
    ]).dropna()
    events = events.astype({"player_id": "int64", "club_id": "int64"})  # consistent types, even for empty inputs
    # Only players who were ever at one of these clubs matter.
    players = events.loc[events["club_id"].isin(club_ids), "player_id"].unique()
    events = events[events["player_id"].isin(players)].sort_values("event_date")

    grid = (pd.MultiIndex.from_product([weeks, players], names=["week", "player_id"])
              .to_frame(index=False).sort_values("week"))
    # Each player's most recent transfer or appearance on or before each Monday.
    at = pd.merge_asof(grid, events, left_on="week", right_on="event_date", by="player_id", direction="backward")
    at = at[at["club_id"].isin(club_ids)]

    vals = valuations[valuations["player_id"].isin(players)][["player_id", "date", "market_value_in_eur"]].copy()
    vals["valued_on"] = pd.to_datetime(vals["date"])
    vals = vals.drop(columns="date").dropna().astype({"player_id": "int64"}).sort_values("valued_on")
    at = pd.merge_asof(at.sort_values("week"), vals, left_on="week", right_on="valued_on",
                       by="player_id", direction="backward")
    at = at[(at["week"] - at["valued_on"]).dt.days <= VALUATION_MAX_DAYS]

    at = at.sort_values(["week", "club_id", "market_value_in_eur"], ascending=[True, True, False])
    best = at.groupby(["week", "club_id"]).head(top)
    out = best.groupby(["week", "club_id"]).agg(value_eur=("market_value_in_eur", "sum"),
                                                players=("player_id", "size")).reset_index()
    return out


def main() -> None:
    print("Reading Transfermarkt tables...", flush=True)
    tm_games, transfers, appearances, valuations = (read(t) for t in
                                                    ("games", "transfers", "appearances", "player_valuations"))

    conn = sqlite3.connect(DB_PATH)
    ours = load_games(conn)[["date", "home", "away", "home_score", "away_score"]]
    mapping = club_mapping(tm_games, ours)
    print(f"Linked {len(mapping)} Transfermarkt clubs to your team names by matching games.")

    weeks = weekly_dates(FIRST_WEEK, valuations["date"].max())
    print(f"Building squad values for {len(weeks)} weeks, {weeks[0]:%Y-%m-%d} to {weeks[-1]:%Y-%m-%d}...", flush=True)
    values = squad_values(transfers, appearances, valuations, set(mapping), weeks)
    values["team"] = values["club_id"].map(mapping)
    values["week"] = values["week"].dt.strftime("%Y-%m-%d")

    with conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS squad_values (
                            league_code TEXT NOT NULL,
                            team        TEXT NOT NULL,
                            week        TEXT NOT NULL,   -- a Monday, 'YYYY-MM-DD'
                            value_eur   REAL NOT NULL,   -- total of the TOP_PLAYERS most valuable players
                            players     INTEGER NOT NULL,
                            PRIMARY KEY (league_code, team, week))""")
        conn.execute("DELETE FROM squad_values WHERE league_code = ?", (LEAGUE_CODE,))
        conn.executemany("INSERT INTO squad_values VALUES (?, ?, ?, ?, ?)",
                         [(LEAGUE_CODE, r.team, r.week, float(r.value_eur), int(r.players))
                          for r in values.itertuples()])
    print(f"Saved {len(values):,} weekly squad values.\n")

    # Coverage: how many Premier League games have a squad value for both teams that week?
    covered = ours[ours["date"] >= FIRST_WEEK].copy()
    dates = pd.to_datetime(covered["date"])
    covered["week"] = dates.dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    covered["season"] = np.where(dates.dt.month >= 7, dates.dt.year, dates.dt.year - 1)
    have = set(zip(values["team"], values["week"]))
    covered["both"] = [(h, w) in have and (a, w) in have
                       for h, a, w in zip(covered["home"], covered["away"], covered["week"])]
    print("Share of games with a squad value for both teams:")
    for season, share in covered.groupby("season")["both"].mean().items():
        print(f"  {season}/{(season + 1) % 100:02d}  {share:.0%}")

    # Sanity check: the most and least valuable squads in the first week of a few seasons.
    for season in (2016, 2020, 2024):
        in_season = set(covered.loc[covered["season"] == season, "home"])
        week_values = values[(values["week"] >= f"{season}-08-08") & values["team"].isin(in_season)]
        if week_values.empty:
            continue
        first = week_values["week"].min()
        week = week_values[week_values["week"] == first].sort_values("value_eur", ascending=False)
        fmt = lambda rows: ", ".join(f"{r.team} €{r.value_eur / 1e6:,.0f}m" for r in rows.itertuples())
        print(f"\nWeek of {first}\n  most valuable:  {fmt(week.head(4))}\n  least valuable: {fmt(week.tail(3))}")
    conn.close()


if __name__ == "__main__":
    main()
