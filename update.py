"""Run the whole weekly update in the right order, stopping at the first failure.

Usage:
    python update.py            # update data and predictions
    python update.py --build    # ...and rebuild the static site in docs/
    python update.py --record   # ...and update the official prediction record (the automated run uses this)

Steps (Premier League, in the epl/ folder):
    1. import_epl          new results, stats and odds from football-data.co.uk
    2. import_understat    expected goals (xG) for the new results
    3. run_backtest        backtest predictions, now including the latest matches
    4. import_fixtures     the next round's fixtures and pre-match odds
    5. predict_upcoming    predictions for those fixtures, saved before kickoff
    6. fetch_availability  a snapshot of player injuries and availability (collected for later use)

Each step is a module run with "python -m", e.g. python -m epl.import_epl.
The NFL joins the update once its model is ready.

Run it before the first kickoff of each round (for weekend matches, Friday evening,
after football-data.co.uk posts the fixtures).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("epl.import_epl", "Results, stats and odds"),
    ("epl.import_understat", "Expected goals (xG)"),
    ("epl.run_backtest", "Backtest predictions"),
    ("epl.import_fixtures", "Upcoming fixtures and odds"),
    ("epl.predict_upcoming", "Predictions for upcoming matches"),
    ("epl.fetch_availability", "Player availability snapshot"),
]
BUILD_STEP = ("build_site", "Static website (docs/)")


def run(module: str, label: str, number: int, total: int, args: list[str]) -> float:
    """Run one step with this project's Python, showing its output. Returns seconds taken."""
    print(f"\n[{number}/{total}] {label}  (python -m {module})")
    print("-" * 60, flush=True)
    started = time.time()
    # sys.executable is the Python running this script, i.e. the one in .venv
    result = subprocess.run([sys.executable, "-m", module, *args])
    elapsed = time.time() - started
    if result.returncode != 0:
        print("-" * 60)
        print(f"\nStopped: {module} failed (exit code {result.returncode}) after {elapsed:.0f}s.")
        print("Nothing after this step was run. Fix the error shown above, then run update.py again.")
        print("Every step is safe to re-run, so starting over from the beginning is fine.")
        sys.exit(1)
    return elapsed


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)  # works no matter which folder it's run from
    steps = STEPS + ([BUILD_STEP] if "--build" in sys.argv[1:] else [])

    started = time.time()
    record = ["--record"] if "--record" in sys.argv[1:] else []
    timings = [
        (label, run(module, label, i, len(steps), record if module == "epl.predict_upcoming" else []))
        for i, (module, label) in enumerate(steps, start=1)
    ]

    print("\n" + "=" * 60)
    print("Update complete.\n")
    for label, seconds in timings:
        print(f"  {label:<36} {seconds:>5.0f}s")
    print(f"  {'Total':<36} {time.time() - started:>5.0f}s")
    if "--build" not in sys.argv[1:]:
        print("\nPreview the site with:  uvicorn app:app --reload")


if __name__ == "__main__":
    main()
