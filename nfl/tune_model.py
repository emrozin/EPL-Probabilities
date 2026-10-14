"""Tune the NFL points model on 2010-2017, then compare everything once on 2018-2025.

Usage:
    python -m nfl.tune_model      # takes about a minute

The points model is tuned (recency half-life, shrinkage, and the margin spread used to turn an
expected margin into a win probability) by walk-forward log loss on the tuning seasons, with
and without the quarterback adjustment. The best settings of each are then scored on the
held-out seasons, next to Elo, the spread and the market. Ties are left out of every score.
"""

import itertools
import sqlite3
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from db import DB_PATH
from nfl.elo import load_nfl_games, load_nfl_market, run_elo, score
from nfl.model import TRAINING_YEARS, add_qb_changes, backtest

TUNE = (2010, 2017)
TEST = (2018, 2025)
HALF_LIVES = [150, 180, 240]
SHRINKAGES = [3, 5, 7]
MARGIN_SDS = [10, 11, 12, 13, 14]
SPREAD_SD = 13.5


def win_loss(games: pd.DataFrame, seasons) -> pd.DataFrame:
    first, last = seasons
    df = games[(games["status"] == "final") & games["season"].between(first, last)
               & (games["home_score"] != games["away_score"])].copy()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    return df


def best_margin_sd(df: pd.DataFrame) -> tuple[float, float]:
    """The margin spread that scores best, and its log loss (margins don't depend on it, so it's cheap to tune)."""
    scores = {sd: score(df.assign(p=norm.cdf(df["model_margin"] / sd)), "p").mean() for sd in MARGIN_SDS}
    sd = min(scores, key=scores.get)
    return sd, scores[sd]


def tune(games: pd.DataFrame, label: str) -> dict:
    target = win_loss(games, TUNE)
    results = []
    for half_life, shrink in itertools.product(HALF_LIVES, SHRINKAGES):
        bt = backtest(games, TUNE[0], half_life, shrink, 13.0, TRAINING_YEARS)
        sd, loss = best_margin_sd(target.merge(bt, on="id"))
        results.append({"half_life_days": half_life, "shrinkage": shrink, "margin_sd": sd, "log_loss": loss})
    results = pd.DataFrame(results).sort_values("log_loss")
    print(f"\n{label}, best 3 on the tuning seasons:")
    print(results.head(3).round(4).to_string(index=False))
    return results.iloc[0].to_dict()


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    games = add_qb_changes(load_nfl_games(conn))
    market = load_nfl_market(conn)
    conn.close()
    no_qb = games.assign(home_qb_change=0, away_qb_change=0)   # same model, QB adjustment switched off

    started = time.time()
    print(f"Tuning on {TUNE[0]}-{TUNE[1]} ({2 * len(HALF_LIVES) * len(SHRINKAGES)} backtests)...", flush=True)
    plain = tune(no_qb, "Points model")
    with_qb = tune(games, "Points model + QB adjustment")
    print(f"\n({time.time() - started:.0f}s)")

    test = win_loss(games, TEST)
    for name, frame, s in (("plain", no_qb, plain), ("qb", games, with_qb)):
        bt = backtest(frame, TEST[0], s["half_life_days"], s["shrinkage"], s["margin_sd"], TRAINING_YEARS)
        test = test.merge(bt[["id", "model_home"]].rename(columns={"model_home": f"{name}_home"}), on="id")
    elo, _, _ = run_elo(games)
    test = test.merge(elo[["id", "elo_home"]], on="id").merge(market, on="id", how="left")
    test["spread_home"] = norm.cdf(test["spread_line"] / SPREAD_SD)

    print(f"\nHeld-out test, {TEST[0]}-{TEST[1]} ({len(test)} games), log loss:")
    rows = [("Coin flip", 0.6931), ("Elo", score(test, "elo_home").mean()),
            ("Points model", score(test, "plain_home").mean()),
            ("Points model + QB adjustment", score(test, "qb_home").mean()),
            ("Spread only", score(test, "spread_home").mean())]
    priced = test.dropna(subset=["market_home"])
    rows.append(("Market moneylines", score(priced, "market_home").mean()))
    for label, value in rows:
        print(f"  {label:<32}{value:.4f}")

    coin, best, market_loss = 0.6931, rows[3][1], rows[5][1]
    print(f"\nShare of the market's edge over a coin flip captured: Elo {(coin - rows[1][1]) / (coin - market_loss):.0%}, "
          f"points + QB {(coin - best) / (coin - market_loss):.0%}")

    changed = test[(test["home_qb_change"] + test["away_qb_change"]) > 0]
    same = test[(test["home_qb_change"] + test["away_qb_change"]) == 0]
    print("\nWhere the QB adjustment helps (held-out):")
    for label, part in (("a team starts a QB who isn't its usual starter", changed), ("both usual starters", same)):
        print(f"  {label:<48}{len(part):>5} games   points model {score(part, 'plain_home').mean():.4f}"
              f"   + QB {score(part, 'qb_home').mean():.4f}   market {score(part.dropna(subset=['market_home']), 'market_home').mean():.4f}")


if __name__ == "__main__":
    main()
