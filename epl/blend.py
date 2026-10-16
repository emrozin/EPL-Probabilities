"""Tune the Premier League model-market blend (see blend.py): python -m epl.blend

Uses the backtest predictions (made from earlier matches only) and the closing odds (Pinnacle,
else Betfair Exchange, margin removed) for home win, draw and away win. The weight is chosen on
the tuning seasons (2016/17-2020/21) and tested on the held-out ones (2021/22 on); the result is
saved in epl/blend.json. The website shows the blend only if it beat the market on the held-out
seasons; either way, the backtest page reports the result.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import blend
from db import DB_PATH
from epl.model import load_market
from epl.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM

BLEND_PATH = Path(__file__).with_name("blend.json")
OUTCOMES = ["home", "draw", "away"]


def load_info() -> dict | None:
    return blend.load(BLEND_PATH)


def probabilities(model: dict, market: dict, w: float) -> dict:
    """The blend for one match, from {'home','draw','away'} dicts of probabilities (0-1)."""
    p = blend.pool([[model[o] for o in OUTCOMES]], [[market[o] for o in OUTCOMES]], w)[0]
    return dict(zip(OUTCOMES, p))


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT g.id, g.season, g.home_score, g.away_score,
               MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
               MAX(CASE WHEN p.outcome = 'draw' THEN p.probability END) AS model_draw,
               MAX(CASE WHEN p.outcome = 'away' THEN p.probability END) AS model_away
        FROM games g JOIN leagues l ON l.id = g.league_id AND l.code = 'EPL'
        JOIN predictions p ON p.game_id = g.id AND p.model_name = ? AND p.model_version = ?
        WHERE g.status = 'final'
        GROUP BY g.id""", conn, params=(BACKTEST_NAME, BACKTEST_VERSION)).merge(load_market(conn), on="id")
    conn.close()
    if df.empty:
        raise SystemExit("No backtest predictions with closing odds. Run python -m epl.run_backtest first.")
    outcome = np.select([df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]], [0, 1], 2)
    model = df[["model_home", "model_draw", "model_away"]].to_numpy()
    market = df[["mkt_home", "mkt_draw", "mkt_away"]].to_numpy()
    tuning = (df["season"] < HELD_OUT_FROM).to_numpy()
    held = ~tuning
    if not tuning.any() or not held.any():
        raise SystemExit("Need backtest predictions in both the tuning and held-out seasons.")
    w, scores = blend.tune(model[tuning], market[tuning], outcome[tuning])
    result = {"weight": w, "tuning_seasons": f"{df['season'].min()}-{HELD_OUT_FROM - 1}",
              "held_out_seasons": f"{HELD_OUT_FROM}-{df['season'].max()}", "held_out_games": int(held.sum()),
              "held_out": {"model": round(blend.log_loss(model[held], outcome[held]), 4),
                           "market": round(blend.log_loss(market[held], outcome[held]), 4),
                           "blend": round(blend.log_loss(blend.pool(model[held], market[held], w), outcome[held]), 4)},
              "tuning_curve": [[wt, round(s, 4)] for wt, s in scores]}
    blend.save(BLEND_PATH, result)
    print("Tuning seasons, log loss by model weight:")
    for wt, s in scores:
        print(f"  {wt:.2f}  {s:.4f}{'  <- best' if wt == w else ''}")
    print(f"Held-out ({result['held_out_games']} matches): {result['held_out']}")
    better = result["held_out"]["blend"] < result["held_out"]["market"]
    print("The blend beat the market on the held-out seasons: the website will show it." if better else
          "The blend didn't beat the market on the held-out seasons: the website won't show it.")
    print(f"Saved to {BLEND_PATH}")


if __name__ == "__main__":
    main()
