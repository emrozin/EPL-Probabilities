"""Does squad value improve the model? Tune its weight on 2016/17-2020/21, then test on 2021/22-2025/26.

Usage:
    python -m epl.import_transfermarkt            # once, to build Transfermarkt squad values
    python -m epl.tune_squad_value                # test them (takes a few minutes: about 20 backtests)

    python -m epl.import_fpl_values                         # once, to build FPL squad values
    python -m epl.tune_squad_value --source fpl             # test FPL prices as they change during the season
    python -m epl.tune_squad_value --source fpl_preseason   # test each season's starting FPL prices

value_weight 0 is today's model. For each combination of value_weight and shrinkage, a
walk-forward backtest is scored on the tuning seasons; the best combination is then compared
with today's settings on the held-out seasons, which played no part in choosing it.
This season (2026/27) is left out: there are no squad values for it yet.
"""

import sqlite3
import sys
import time

import numpy as np
import pandas as pd

from epl.model import SHRINKAGE, add_scores, backtest, load_games, load_market, load_squad_values

DB_PATH = "data/sports.db"
# FPL prices span a much narrower range than market values (roughly 1.4x from the cheapest squad
# to the priciest, against about 20x), so the same effect needs a much larger weight.
VALUE_WEIGHTS = {"transfermarkt": [0, 0.1, 0.2, 0.35, 0.5], "fpl": [0, 1, 2, 3, 4, 6],
                 "fpl_preseason": [0, 1, 2, 3, 4, 6]}
SHRINKAGES = [0.5, 1, 2, 4]
TUNE = (2016, 2020)
TEST = (2021, 2025)


def scored_backtest(games, seasons, values, **settings) -> pd.DataFrame:
    first, last = seasons
    results = backtest(games[games["season"] <= last], start_season=first,
                       squad_values_by_week=values, **settings)
    add_scores(results, "model")
    return results


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    source = argv[argv.index("--source") + 1] if "--source" in argv else "transfermarkt"
    if source not in VALUE_WEIGHTS:
        raise SystemExit(f"Unknown source {source!r}: choose one of {', '.join(VALUE_WEIGHTS)}")
    weights = VALUE_WEIGHTS[source]

    conn = sqlite3.connect(DB_PATH)
    games = load_games(conn)
    values = load_squad_values(conn, source=source)
    market = load_market(conn, closing=True)
    conn.close()
    if not values:
        script = "python -m epl.import_fpl_values" if source.startswith("fpl") else "python -m epl.import_transfermarkt"
        raise SystemExit(f"No {source} squad values found. Run {script} first.")
    print(f"Squad values from: {source}")

    print(f"Tuning on {TUNE[0]}/{(TUNE[0] + 1) % 100:02d}-{TUNE[1]}/{(TUNE[1] + 1) % 100:02d} "
          f"({len(weights) * len(SHRINKAGES)} combinations)...\n")
    grid = {}
    started = time.time()
    for weight in weights:
        for shrink in SHRINKAGES:
            res = scored_backtest(games, TUNE, values, value_weight=weight, shrinkage=shrink)
            grid[(weight, shrink)] = res["model_logloss"].mean()
            print(f"  value_weight {weight:<5} shrinkage {shrink:<4} log loss {grid[(weight, shrink)]:.4f}"
                  f"   ({time.time() - started:.0f}s)", flush=True)

    table = pd.Series(grid).unstack()
    table.index.name, table.columns.name = "value_weight", "shrinkage"
    print("\nTuning seasons, log loss (lower is better):")
    print(table.round(4).to_string())
    best_weight, best_shrink = min(grid, key=grid.get)
    print(f"\nBest: value_weight {best_weight}, shrinkage {best_shrink} "
          f"(today's model, value_weight 0 and shrinkage {SHRINKAGE}: {grid[(0, SHRINKAGE)]:.4f})")

    print(f"\nHeld-out test, {TEST[0]}/{(TEST[0] + 1) % 100:02d}-{TEST[1]}/{(TEST[1] + 1) % 100:02d}...", flush=True)
    today = scored_backtest(games, TEST, values, value_weight=0, shrinkage=SHRINKAGE)
    best = scored_backtest(games, TEST, values, value_weight=best_weight, shrinkage=best_shrink)
    both = (today[["id", "season", "result", "model_logloss"]]
            .merge(best[["id", "model_logloss"]], on="id", suffixes=("_today", "_value"))
            .merge(market, on="id"))
    add_scores(both, "mkt")
    both = both.sort_values("id")
    rounds = both.groupby("season").cumcount() // 10 + 1   # approximate round of the season

    def line(label, part):
        print(f"  {label:<18}{len(part):>7}{part['model_logloss_today'].mean():>10.4f}"
              f"{part['model_logloss_value'].mean():>13.4f}{part['mkt_logloss'].mean():>9.4f}")

    print(f"  {'':<18}{'matches':>7}{'today':>10}{'with value':>13}{'market':>9}")
    line("all matches", both)
    line("rounds 1-10", both[rounds <= 10])
    line("rounds 11-38", both[rounds > 10])
    gap_today = both["model_logloss_today"].mean() - both["mkt_logloss"].mean()
    gap_value = both["model_logloss_value"].mean() - both["mkt_logloss"].mean()
    if gap_today > 0:
        print(f"\nShare of the gap to the market closed by squad value: {(gap_today - gap_value) / gap_today:.0%}")


if __name__ == "__main__":
    main()
