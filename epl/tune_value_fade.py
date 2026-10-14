"""Does fading squad value out during the season beat using it at full strength all season?

Usage:
    python -m epl.tune_value_fade                          # FPL preseason prices (usable for live predictions)
    python -m epl.tune_value_fade --source fpl             # FPL prices as they change during the season
    python -m epl.tune_value_fade --source transfermarkt   # for comparison only: no data after June 2026

tune_squad_value.py found squad value helps early in a season and barely at all later. Here the
squad-value adjustment halves every `fade` rounds (see model.faded_settings), so late in the season
the model becomes today's model again. "no fade" is the full-strength version tested before.

As before: every combination is scored on 2016/17-2020/21, the best one is chosen there, and only
then are the best "no fade" and the best "fade" versions checked once on 2021/22-2025/26.
Takes roughly 10-15 minutes.
"""

import sqlite3
import sys
import time

import pandas as pd

from epl.model import SHRINKAGE, add_scores, backtest, load_games, load_market, load_squad_values

DB_PATH = "data/sports.db"
# Squad-value weights around the ones that did best in tune_squad_value.py. FPL prices span a much
# narrower range than market values, so the same effect needs a much larger weight.
VALUE_WEIGHTS = {"transfermarkt": [0.1, 0.2, 0.35, 0.5], "fpl": [1, 2, 3, 4, 6],
                 "fpl_preseason": [1, 2, 3, 4, 6]}
SHRINKAGES = [1, 2, 4]
FADES = [None, 3, 6, 12]   # rounds for the adjustment to halve; None = full strength all season
TUNE = (2016, 2020)
TEST = (2021, 2025)


def season_label(seasons) -> str:
    first, last = seasons
    return f"{first}/{(first + 1) % 100:02d}-{last}/{(last + 1) % 100:02d}"


def scored_backtest(games, seasons, values, **settings) -> pd.DataFrame:
    first, last = seasons
    results = backtest(games[games["season"] <= last], start_season=first,
                       squad_values_by_week=values, **settings)
    add_scores(results, "model")
    return results[["id", "season", "result", "model_logloss"]]


def settings(key) -> dict:
    weight, shrink, fade = key
    return {"value_weight": weight, "shrinkage": shrink, "value_fade_rounds": fade}


def describe(key) -> str:
    weight, shrink, fade = key
    return f"value_weight {weight}, shrinkage {shrink}, " + ("no fade" if fade is None else f"halves every {fade} rounds")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    source = argv[argv.index("--source") + 1] if "--source" in argv else "fpl_preseason"
    if source not in VALUE_WEIGHTS:
        raise SystemExit(f"Unknown source {source!r}: choose one of {', '.join(VALUE_WEIGHTS)}")

    conn = sqlite3.connect(DB_PATH)
    games = load_games(conn)
    values = load_squad_values(conn, source=source)
    market = load_market(conn, closing=True)
    conn.close()
    if not values:
        script = "python -m epl.import_fpl_values" if source.startswith("fpl") else "python -m epl.import_transfermarkt"
        raise SystemExit(f"No {source} squad values found. Run {script} first.")

    combos = [(w, s, f) for f in FADES for w in VALUE_WEIGHTS[source] for s in SHRINKAGES]
    print(f"Squad values from: {source}")
    print(f"Tuning on {season_label(TUNE)} ({len(combos) + 1} backtests)...\n")
    started = time.time()
    today_key = (0, SHRINKAGE, None)
    grid = {today_key: scored_backtest(games, TUNE, values, **settings(today_key))["model_logloss"].mean()}
    print(f"  today's model{'':<38} log loss {grid[today_key]:.4f}   ({time.time() - started:.0f}s)", flush=True)
    for key in combos:
        grid[key] = scored_backtest(games, TUNE, values, **settings(key))["model_logloss"].mean()
        print(f"  {describe(key):<51} log loss {grid[key]:.4f}   ({time.time() - started:.0f}s)", flush=True)

    best_by_fade = {}
    for fade in FADES:
        key = min((k for k in combos if k[2] == fade), key=grid.get)
        best_by_fade[fade] = key
    print("\nBest on the tuning seasons, for each fade speed (lower is better):")
    print(f"  {'today':<10}{grid[today_key]:.4f}")
    for fade, key in best_by_fade.items():
        label = "no fade" if fade is None else f"{fade} rounds"
        print(f"  {label:<10}{grid[key]:.4f}   ({describe(key)})")

    best_fixed = best_by_fade[None]
    best_faded = min((best_by_fade[f] for f in FADES if f is not None), key=grid.get)
    print(f"\nChosen for the held-out test, without looking at it:\n"
          f"  no fade: {describe(best_fixed)}\n  fade:    {describe(best_faded)}")

    print(f"\nHeld-out test, {season_label(TEST)}...", flush=True)
    columns = {"today": today_key, "no fade": best_fixed, "fade": best_faded}
    both = None
    for name, key in columns.items():
        res = scored_backtest(games, TEST, values, **settings(key)).rename(columns={"model_logloss": name})
        both = res if both is None else both.merge(res[["id", name]], on="id")
    both = both.merge(market, on="id")
    add_scores(both, "mkt")
    both = both.sort_values("id")
    rounds = both.groupby("season").cumcount() // 10 + 1   # approximate round of the season

    def line(label, part):
        cells = "".join(f"{part[name].mean():>10.4f}" for name in columns)
        print(f"  {label:<14}{len(part):>8}{cells}{part['mkt_logloss'].mean():>10.4f}")

    print(f"  {'':<14}{'matches':>8}" + "".join(f"{name:>10}" for name in columns) + f"{'market':>10}")
    line("all matches", both)
    line("rounds 1-10", both[rounds <= 10])
    line("rounds 11-38", both[rounds > 10])

    gap = both["today"].mean() - both["mkt_logloss"].mean()
    if gap > 0:
        for name in ("no fade", "fade"):
            print(f"Share of the gap to the market closed ({name}): {(both['today'].mean() - both[name].mean()) / gap:.0%}")


if __name__ == "__main__":
    main()
