"""How well does the model handle newly promoted teams? Run before (and after) adding Championship data.

Usage:
    python import_epl.py --league CHAMP   # not needed for this check, but the next step uses it
    python run_backtest.py                # make sure the backtest predictions are current
    python promoted_teams.py

Uses the saved backtest predictions (2016/17 onward) and the closing market to show:
  1. the model's log loss vs the market's on promoted teams' matches, split by how many
     Premier League games the promoted team had played that season;
  2. points the model and the market expected promoted teams to win in their first 10
     matches, against the points they actually won.
"""

import sqlite3

import numpy as np
import pandas as pd

from model import add_scores, load_games, load_market
from run_backtest import BACKTEST_NAME, BACKTEST_VERSION

DB_PATH = "data/sports.db"
STAGES = [(1, 5), (6, 10), (11, 19), (20, 38)]  # a promoted team's nth Premier League match of the season


def promoted_teams(games: pd.DataFrame) -> set[tuple[int, str]]:
    """(season, team) for every team that wasn't in the league the season before."""
    teams_by_season = {s: set(g["home"]) | set(g["away"]) for s, g in games.groupby("season")}
    return {(s, t) for s, teams in teams_by_season.items() if s - 1 in teams_by_season
            for t in teams - teams_by_season[s - 1]}


def team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match: each match appears twice, once from each side."""
    sides = []
    for side, other in (("home", "away"), ("away", "home")):
        sides.append(pd.DataFrame({
            "id": df["id"], "season": df["season"], "date": df["date"], "team": df[side],
            "points": np.select([df["result"] == side, df["result"] == "draw"], [3, 1], 0),
            "model_xpts": 3 * df[f"model_{side}"] + df["model_draw"],
            "market_xpts": 3 * df[f"mkt_{side}"] + df["mkt_draw"],
            "model_logloss": df["model_logloss"], "mkt_logloss": df["mkt_logloss"],
        }))
    rows = pd.concat(sides, ignore_index=True).sort_values(["season", "team", "date"])
    rows["match_number"] = rows.groupby(["season", "team"]).cumcount() + 1
    return rows


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    games = load_games(conn)
    predictions = pd.read_sql("""
        SELECT game_id AS id,
               MAX(CASE WHEN outcome = 'home' THEN probability END) AS model_home,
               MAX(CASE WHEN outcome = 'draw' THEN probability END) AS model_draw,
               MAX(CASE WHEN outcome = 'away' THEN probability END) AS model_away
        FROM predictions WHERE model_name = ? AND model_version = ?
        GROUP BY game_id
    """, conn, params=(BACKTEST_NAME, BACKTEST_VERSION))
    df = games.merge(predictions, on="id").merge(load_market(conn, closing=True), on="id")
    df["result"] = np.select([df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
                             ["home", "draw"], default="away")
    add_scores(df, "model")
    add_scores(df, "mkt")

    promoted = promoted_teams(games)
    rows = team_rows(df)
    rows["promoted"] = [(s, t) in promoted for s, t in zip(rows["season"], rows["team"])]
    others = rows[~rows["promoted"]]
    promo = rows[rows["promoted"]]
    seasons = sorted(df["season"].unique())
    print(f"Backtest matches with market odds: {len(df):,}, seasons {seasons[0]}/{(seasons[0] + 1) % 100:02d} "
          f"to {seasons[-1]}/{(seasons[-1] + 1) % 100:02d}; promoted team-seasons: "
          f"{promo.groupby(['season', 'team']).ngroups}\n")

    print("1. Log loss on promoted teams' matches, by the promoted team's match number (lower is better)")
    print(f"   {'Matches':<14}{'count':>7}{'model':>9}{'market':>9}{'gap':>9}")
    for first, last in STAGES:
        part = promo[promo["match_number"].between(first, last)]
        gap = part["model_logloss"].mean() - part["mkt_logloss"].mean()
        print(f"   {f'{first}-{last}':<14}{len(part):>7}{part['model_logloss'].mean():>9.4f}"
              f"{part['mkt_logloss'].mean():>9.4f}{gap:>+9.4f}")
    gap = others["model_logloss"].mean() - others["mkt_logloss"].mean()
    print(f"   {'other matches':<14}{len(others):>7}{others['model_logloss'].mean():>9.4f}"
          f"{others['mkt_logloss'].mean():>9.4f}{gap:>+9.4f}")
    print("   (a gap above the 'other matches' gap means the model does relatively worse there)\n")

    first10 = promo[promo["match_number"] <= 10]
    per_team = first10.groupby(["season", "team"])[["points", "model_xpts", "market_xpts"]].sum()
    print("2. Points per promoted team in its first 10 matches (average)")
    print(f"   actual {per_team['points'].mean():.1f}   model expected {per_team['model_xpts'].mean():.1f}   "
          f"market expected {per_team['market_xpts'].mean():.1f}")
    print("   (model expecting more than actual means it overrates promoted teams early on)")
    conn.close()


if __name__ == "__main__":
    main()
