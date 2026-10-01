"""Where does the model lose ground to the betting market? A breakdown of the backtest.

Usage:
    python run_backtest.py      # make sure the backtest predictions are current
    python error_analysis.py

Each table compares the model's log loss with the closing market's (lower is better).
"gap" is model minus market per match; "share" is how much of the model's total shortfall
that group accounts for (gap x matches, as a share of the whole), so a group can matter
because it's much worse per match or because it's large.
"""

import sqlite3

import numpy as np
import pandas as pd

from model import add_scores, load_games, load_market
from run_backtest import BACKTEST_NAME, BACKTEST_VERSION

DB_PATH = "data/sports.db"
MIN_TEAM_MATCHES = 60


def load_backtest_frame(conn) -> pd.DataFrame:
    """Backtest predictions joined with results and closing market probabilities, scored."""
    predictions = pd.read_sql("""
        SELECT game_id AS id,
               MAX(CASE WHEN outcome = 'home' THEN probability END) AS model_home,
               MAX(CASE WHEN outcome = 'draw' THEN probability END) AS model_draw,
               MAX(CASE WHEN outcome = 'away' THEN probability END) AS model_away
        FROM predictions WHERE model_name = ? AND model_version = ?
        GROUP BY game_id
    """, conn, params=(BACKTEST_NAME, BACKTEST_VERSION))
    df = load_games(conn).merge(predictions, on="id").merge(load_market(conn, closing=True), on="id")
    df["result"] = np.select([df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
                             ["home", "draw"], default="away")
    add_scores(df, "model")
    add_scores(df, "mkt")
    df["gap"] = df["model_logloss"] - df["mkt_logloss"]
    # Round of the season: each team plays once per round, so a season's matches in date order, ten at a time.
    df = df.sort_values(["season", "date", "home"])
    df["round"] = df.groupby("season").cumcount() // 10 + 1
    return df


def breakdown(df: pd.DataFrame, group: pd.Series, total_gap: float) -> pd.DataFrame:
    out = df.groupby(group, observed=True).agg(matches=("gap", "size"), model=("model_logloss", "mean"),
                                               market=("mkt_logloss", "mean"), gap=("gap", "mean"))
    out["share"] = out["gap"] * out["matches"] / total_gap
    return out


def show(title: str, table: pd.DataFrame, note: str = "") -> None:
    print(f"\n{title}")
    print(f"   {'':<16}{'matches':>8}{'model':>9}{'market':>9}{'gap':>9}{'share':>8}")
    for name, r in table.iterrows():
        share = r.share if abs(r.share) >= 0.005 else 0.0   # avoid showing "-0%"
        print(f"   {str(name):<16}{int(r.matches):>8}{r.model:>9.4f}{r.market:>9.4f}{r.gap:>+9.4f}{share:>8.0%}")
    if note:
        print(f"   {note}")


def calibration(df: pd.DataFrame) -> None:
    print("\n3. Calibration: average predicted chance vs how often it happened")
    print(f"   {'outcome':<10}{'actual':>9}{'model':>9}{'market':>9}")
    for outcome in ("home", "draw", "away"):
        print(f"   {outcome:<10}{(df['result'] == outcome).mean():>9.1%}"
              f"{df[f'model_{outcome}'].mean():>9.1%}{df[f'mkt_{outcome}'].mean():>9.1%}")
    # Within each band of the model's home-win chance, how often did the home team actually win?
    bands = pd.cut(df["model_home"], [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0])
    print(f"\n   Home win, by the model's predicted chance")
    print(f"   {'predicted':<14}{'matches':>8}{'avg predicted':>15}{'actual':>9}")
    for band, part in df.groupby(bands, observed=True):
        print(f"   {str(band):<14}{len(part):>8}{part['model_home'].mean():>15.1%}{(part['result'] == 'home').mean():>9.1%}")


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = load_backtest_frame(conn)
    conn.close()
    total_gap = df["gap"].sum()
    print(f"Backtest matches with closing odds: {len(df):,}. Model log loss {df['model_logloss'].mean():.4f}, "
          f"market {df['mkt_logloss'].mean():.4f}, gap {df['gap'].mean():+.4f} per match.")

    stages = pd.cut(df["round"], [0, 5, 10, 19, 29, 38], labels=["rounds 1-5", "rounds 6-10", "rounds 11-19",
                                                                  "rounds 20-29", "rounds 30-38"])
    show("1. By stage of the season", breakdown(df, stages, total_gap))

    favourite = df[["mkt_home", "mkt_draw", "mkt_away"]].max(axis=1)
    bands = pd.cut(favourite, [0, 0.4, 0.5, 0.6, 0.7, 1.0],
                   labels=["under 40%", "40-50%", "50-60%", "60-70%", "70% or more"])
    show("2. By the market favourite's chance (how one-sided the match is)", breakdown(df, bands, total_gap))

    calibration(df)

    rows = []
    for side in ("home", "away"):
        part = df[[side, "gap", "model_logloss", "mkt_logloss"]].rename(columns={side: "team"})
        rows.append(part)
    teams = pd.concat(rows)
    per_team = breakdown(teams, teams["team"], total_gap * 2)   # every match counted for both teams
    per_team = per_team[per_team["matches"] >= MIN_TEAM_MATCHES].sort_values("gap")
    show(f"4a. Teams the model forecasts best, relative to the market (at least {MIN_TEAM_MATCHES} matches)",
         per_team.head(5))
    show("4b. Teams the model forecasts worst", per_team.tail(5).iloc[::-1],
         "Each match counts for both teams here, so shares are of twice the total.")


if __name__ == "__main__":
    main()
