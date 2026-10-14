"""Tune the NFL Elo model on 2010-2017, then test it once on 2018-2025 against the betting market.

Usage:
    python -m nfl.import_nfl     # once, to load the games
    python -m nfl.tune_elo

Elo plays through every game since 1999, so each prediction only uses earlier results. Every
combination of settings is scored by log loss on the tuning seasons (ties left out, since a
win probability has nothing to say about them); the best one is then checked on the held-out
seasons, which played no part in choosing it.

Benchmarks: the market's moneylines (margin removed), and a probability made from the spread
alone (assuming the actual margin lands within about 13.5 points of it, as it typically does).
"""

import itertools
import sqlite3
import time

import pandas as pd
from scipy.stats import norm

from db import DB_PATH
from nfl.elo import load_nfl_games, load_nfl_market, run_elo, score

TUNE = (2010, 2017)
TEST = (2018, 2025)
GRID = {
    "k": [15, 20, 25, 30, 40],
    "home_advantage": [40, 55, 70],
    "home_learning": [0, 0.02, 0.05, 0.1],
    "revert": [1 / 3, 0.5, 0.6, 0.75],
}
SPREAD_SD = 13.5  # typical gap between the actual margin and the spread, in points


def finished(games: pd.DataFrame, predictions: pd.DataFrame, seasons) -> pd.DataFrame:
    first, last = seasons
    df = games.merge(predictions, on="id")
    df = df[(df["status"] == "final") & df["season"].between(first, last) & (df["home_score"] != df["away_score"])].copy()
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(float)
    return df


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    games = load_nfl_games(conn)
    market = load_nfl_market(conn)
    conn.close()
    if games.empty:
        raise SystemExit("No NFL games found. Run python -m nfl.import_nfl first.")

    combos = [dict(zip(GRID, values)) for values in itertools.product(*GRID.values())]
    print(f"Tuning on {TUNE[0]}-{TUNE[1]} ({len(combos)} combinations)...", flush=True)
    started = time.time()
    results = []
    for settings in combos:
        predictions, _, _ = run_elo(games, **settings)
        tune = finished(games, predictions, TUNE)
        results.append({**settings, "log_loss": score(tune, "elo_home").mean()})
    results = pd.DataFrame(results).sort_values("log_loss")
    print(f"Done in {time.time() - started:.0f}s. Best 5:")
    print(results.head(5).round(4).to_string(index=False))

    best = results.iloc[0][list(GRID)].to_dict()
    default = {"k": 20, "home_advantage": 55, "home_learning": 0, "revert": 1 / 3}   # common textbook settings
    print(f"\nChosen: k {best['k']:.0f}, home advantage {best['home_advantage']:.0f}, "
          f"home learning {best['home_learning']}, revert {best['revert']:.2f}")

    print(f"\nHeld-out test, {TEST[0]}-{TEST[1]}:")
    tuned_pred, ratings, hfa = run_elo(games, **best)
    plain_pred, _, _ = run_elo(games, **default)
    test = finished(games, tuned_pred, TEST).merge(
        plain_pred[["id", "elo_home"]].rename(columns={"elo_home": "plain_home"}), on="id")
    test["spread_home"] = norm.cdf(test["spread_line"] / SPREAD_SD)
    with_market = test.merge(market, on="id")

    rows = [("Coin flip", 0.6931, len(test)),
            ("Elo, textbook settings", score(test, "plain_home").mean(), len(test)),
            ("Elo, tuned", score(test, "elo_home").mean(), len(test)),
            ("Spread only", score(test, "spread_home").mean(), len(test)),
            ("Market moneylines", score(with_market, "market_home").mean(), len(with_market))]
    for label, value, n in rows:
        print(f"  {label:<48}{value:.4f}   ({n} games)")

    gap_plain = rows[1][1] - rows[4][1]
    gap_tuned = rows[2][1] - rows[4][1]
    print(f"\nTuned Elo is {gap_tuned:.4f} behind the market "
          f"(textbook settings: {gap_plain:.4f}).")
    picked = ((test["elo_home"] > 0.5) == (test["home_win"] == 1)).mean()
    print(f"Elo picked the winner in {picked:.1%} of held-out games.")
    print(f"Home advantage now: {hfa:.0f} Elo points (about {hfa / 25:.1f} points of margin).")

    print("\nCurrent ratings (top and bottom 5):")
    table = pd.Series(ratings).sort_values(ascending=False).round(0).astype(int)
    print(pd.concat([table.head(5), table.tail(5)]).to_string())


if __name__ == "__main__":
    main()
