"""Tune the NFL model-market blend (see blend.py): python -m nfl.blend

Uses the backtest predictions (made week by week from earlier games only) and the closing
moneylines, ties left out. The weight is chosen on 2010-2017 and tested on 2018-2025; the
result is saved in nfl/blend.json, which the website and nfl/predict_upcoming.py read.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import blend
from db import DB_PATH
from nfl.elo import load_nfl_games, load_nfl_market
from nfl.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM

BLEND_PATH = Path(__file__).with_name("blend.json")


def load_weight() -> float | None:
    info = blend.load(BLEND_PATH)
    return info["weight"] if info else None


def home_win(model_home: float, market_home: float, w: float) -> float:
    """The blend's home win probability for one game."""
    return float(blend.pool([[model_home, 1 - model_home]], [[market_home, 1 - market_home]], w)[0, 0])


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    games = load_nfl_games(conn)
    preds = pd.read_sql("""SELECT game_id AS id, probability AS model_home FROM predictions
                           WHERE model_name = ? AND model_version = ? AND outcome = 'home'""",
                        conn, params=(BACKTEST_NAME, BACKTEST_VERSION))
    df = (games[games["status"] == "final"].merge(preds, on="id").merge(load_nfl_market(conn), on="id")
          .query("home_score != away_score and season <= 2025"))   # complete seasons only
    conn.close()
    two = lambda p: np.column_stack([p, 1 - p])
    outcome = np.where(df["home_score"] > df["away_score"], 0, 1)
    tuning = (df["season"] < HELD_OUT_FROM).to_numpy()
    w, scores = blend.tune(two(df["model_home"][tuning]), two(df["market_home"][tuning]), outcome[tuning])
    held = ~tuning
    m, k = two(df["model_home"][held]), two(df["market_home"][held])
    result = {"weight": w, "tuning_seasons": f"{df['season'].min()}-{HELD_OUT_FROM - 1}",
              "held_out_seasons": f"{HELD_OUT_FROM}-{df['season'].max()}", "held_out_games": int(held.sum()),
              "held_out": {"model": round(blend.log_loss(m, outcome[held]), 4),
                           "market": round(blend.log_loss(k, outcome[held]), 4),
                           "blend": round(blend.log_loss(blend.pool(m, k, w), outcome[held]), 4)},
              "tuning_curve": [[wt, round(s, 4)] for wt, s in scores]}
    blend.save(BLEND_PATH, result)
    print("Tuning seasons, log loss by model weight:")
    for wt, s in scores:
        print(f"  {wt:.2f}  {s:.4f}{'  <- best' if wt == w else ''}")
    print(f"Held-out {result['held_out_seasons']} ({result['held_out_games']} games): {result['held_out']}")
    print(f"Saved to {BLEND_PATH}")


if __name__ == "__main__":
    main()
