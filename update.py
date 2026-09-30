"""Run the whole weekly update in the right order, stopping at the first failure.

Usage:
    python update.py            # update data and predictions
    python update.py --build    # ...and rebuild the static site in docs/

Steps:
    1. import_epl.py        new results, stats and odds from football-data.co.uk
    2. import_understat.py  expected goals (xG) for the new results
    3. run_backtest.py      backtest predictions, now including the latest matches
    4. import_fixtures.py   the next round's fixtures and pre-match odds
    5. predict_upcoming.py  predictions for those fixtures, saved before kickoff

Run it before the first kickoff of each round (for weekend matches, Friday evening,
after football-data.co.uk posts the fixtures).
"""

import os
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    ("import_epl.py", "Results, stats and odds"),
    ("import_understat.py", "Expected goals (xG)"),
    ("run_backtest.py", "Backtest predictions"),
    ("import_fixtures.py", "Upcoming fixtures and odds"),
    ("predict_upcoming.py", "Predictions for upcoming matches"),
]
BUILD_STEP = ("build_site.py", "Static website (docs/)")


def run(script: str, label: str, number: int, total: int) -> float:
    """Run one step with this project's Python, showing its output. Returns seconds taken."""
    print(f"\n[{number}/{total}] {label}  ({script})")
    print("-" * 60, flush=True)
    started = time.time()
    # sys.executable is the Python running this script, i.e. the one in .venv
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - started
    if result.returncode != 0:
        print("-" * 60)
        print(f"\nStopped: {script} failed (exit code {result.returncode}) after {elapsed:.0f}s.")
        print("Nothing after this step was run. Fix the error shown above, then run update.py again.")
        print("Every step is safe to re-run, so starting over from the beginning is fine.")
        sys.exit(1)
    return elapsed


def main() -> None:
    os.chdir(Path(__file__).resolve().parent)  # works no matter which folder it's run from
    steps = STEPS + ([BUILD_STEP] if "--build" in sys.argv[1:] else [])

    started = time.time()
    timings = [(label, run(script, label, i, len(steps))) for i, (script, label) in enumerate(steps, start=1)]

    print("\n" + "=" * 60)
    print("Update complete.\n")
    for label, seconds in timings:
        print(f"  {label:<36} {seconds:>5.0f}s")
    print(f"  {'Total':<36} {time.time() - started:>5.0f}s")
    if "--build" not in sys.argv[1:]:
        print("\nPreview the site with:  uvicorn app:app --reload")


if __name__ == "__main__":
    main()
