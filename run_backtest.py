"""Run the walk-forward backtest and save its predictions to the database.

Usage:
    python run_backtest.py

For every week since 2016/17, the model is trained only on matches played before
that week and predicts that week's games. The predictions are saved under their
own model name, separate from the live predictions made by predict_upcoming.py.
The website's backtest and results pages read them. Re-run after importing new
results so the backtest covers the latest matches (it takes a few seconds).
"""

import sqlite3

from model import backtest, load_games

DB_PATH = "data/sports.db"
BACKTEST_NAME = "poisson_xg_backtest"
BACKTEST_VERSION = "v2"
START_SEASON = 2016
HELD_OUT_FROM = 2021  # settings were tuned on 2016/17-2020/21; 2021/22 onward is the honest test


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    games = load_games(conn)
    results = backtest(games, start_season=START_SEASON)

    rows = [
        (r.id, BACKTEST_NAME, BACKTEST_VERSION, outcome, float(getattr(r, f"model_{outcome}")))
        for r in results.itertuples()
        for outcome in ("home", "draw", "away")
    ]
    # Replace the previous backtest run entirely, in one transaction.
    with conn:
        conn.execute("DELETE FROM predictions WHERE model_name = ? AND model_version = ?",
                     (BACKTEST_NAME, BACKTEST_VERSION))
        conn.executemany(
            """
            INSERT INTO predictions (game_id, model_name, model_version, outcome, probability)
            VALUES (?, ?, ?, ?, ?)
            """,
            rows,
        )
    conn.close()
    print(f"Saved backtest predictions for {len(results)} matches since {START_SEASON}/{(START_SEASON + 1) % 100:02d}.")


if __name__ == "__main__":
    main()
