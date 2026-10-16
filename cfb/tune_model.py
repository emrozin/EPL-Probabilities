"""Tune the college football model's settings on 2010-2017: python -m cfb.tune_model

A grid of recency (half-life), the weight of earlier seasons, shrinkage and training years; for
each, the walk-forward backtest's log loss (ties left out) at several margin spreads. Only the
tuning seasons are used; the held-out seasons (2018 on) are scored once in cfb/run_backtest.py.
"""

import itertools
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm

from cfb.model import backtest, load_games
from db import DB_PATH

TUNING = (2010, 2017)
MARGIN_SDS = [13, 14, 15, 15.5, 16, 17]
GRID = {"half_life": [300, 400], "season_weight": [0.3, 0.4, 0.5], "shrinkage": [0.05, 0.1, 0.25],
        "years": [4]}
# A first, wider grid (half-life 100-400, season weight 0.4-1, shrinkage 1-8, 2-4 years) put the
# best settings at its edges (half-life 400, shrinkage 1, 4 years); a second (half-life 400-1200,
# season weight 0.5-0.8, shrinkage 0.25-1, 4-5 years) at shrinkage 0.25 and season weight 0.5, so this
# third one extends past those.


def score(results: pd.DataFrame, games: pd.DataFrame) -> dict:
    d = results.merge(games[["id", "home_score", "away_score"]], on="id")
    d = d[d["home_score"] != d["away_score"]]
    won = (d["home_score"] > d["away_score"]).to_numpy()
    out = {}
    for sd in MARGIN_SDS:
        p = np.clip(norm.cdf(d["model_margin"] / sd), 1e-6, 1 - 1e-6)
        out[sd] = float(-np.mean(np.where(won, np.log(p), np.log(1 - p))))
    return out


def main() -> None:
    games = load_games(sqlite3.connect(DB_PATH))
    rows = []
    grid = itertools.product(GRID["half_life"], GRID["season_weight"], GRID["shrinkage"], GRID["years"])
    for half_life, season_weight, shrinkage, years in grid:
        results = backtest(games, *TUNING, half_life_days=half_life, season_weight=season_weight,
                           shrinkage=shrinkage, training_years=years)
        for sd, loss in score(results, games).items():
            rows.append({"half_life": half_life, "season_weight": season_weight, "shrinkage": shrinkage,
                         "years": years, "margin_sd": sd, "log_loss": loss})
        best = min(rows, key=lambda r: r["log_loss"])
        print(f"half-life {half_life}, season weight {season_weight}, shrinkage {shrinkage}, {years} years: "
              f"best so far {best['log_loss']:.4f}", flush=True)
    table = pd.DataFrame(rows).sort_values("log_loss")
    print(table.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
