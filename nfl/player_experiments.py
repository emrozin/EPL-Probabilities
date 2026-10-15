"""Experiments: which extra information improves the NFL player projections?

Usage:
    python -m nfl.player_experiments            # tuning seasons (about 5 minutes)
    python -m nfl.player_experiments --held-out # also the held-out seasons, for the versions kept

Each version is scored with total_error (nfl/players.py): the average miss per player-game actually
played, counting every player who was projected or played, so a version can't look better just by
projecting fewer players. Tuning seasons are 2013-2017 (snap counts start in 2013); the held-out
check is 2018-2025.

Results (October 2026), receiving yards miss on the tuning seasons, lower is better:
    no injury reports                         20.39   (2012-2017)
    injury reports (Out / Doubtful removed)   19.83   kept; held-out 19.94 -> 19.14
    + a ruled-out player's share redistributed 19.95  not kept
    + snap counts                             19.82   kept; held-out 19.14 -> 19.01
    + defense by position                     19.97   not kept
    + air yards (depth of target)             20.30   not kept
"""

import sqlite3
import sys
from dataclasses import replace

from db import DB_PATH
from nfl.players import SETTINGS, Projector, backtest_targets, load, load_injuries, load_snaps, total_error

VERSIONS = {
    "no injury reports or snap counts": (replace(SETTINGS, snaps=0.0), False),
    "injury reports, no snap counts": (replace(SETTINGS, snaps=0.0), True),
    "current: injury reports + snap counts": (SETTINGS, True),
    "+ redistributing ruled-out players' shares": (replace(SETTINGS, redistribute=1.0), True),
    "+ defense by position": (replace(SETTINGS, defense_by_position=True), True),
    "+ air yards": (replace(SETTINGS, air_yards=0.5), True),
}
KEPT = ["no injury reports or snap counts", "current: injury reports + snap counts"]


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    conn = sqlite3.connect(DB_PATH)
    games, teams, players = load(conn)
    injuries, snaps = load_injuries(conn), load_snaps(conn)
    periods = [("Tuning seasons 2013-2017", (2013, 2017), list(VERSIONS))]
    if "--held-out" in argv:
        periods.append(("Held-out seasons 2018-2025", (2018, 2025), KEPT))
    for label, (first, last), names in periods:
        targets = backtest_targets(conn, first, last)
        print(f"\n{label}: average miss per player-game played (lower is better)")
        print(f"  {'':<46}{'pass yds':>9}{'rush yds':>9}{'rec':>7}{'rec yds':>9}")
        for name in names:
            settings, use_injuries = VERSIONS[name]
            proj = Projector(games, teams, players, settings, injuries if use_injuries else None, snaps).run(targets)
            e = total_error(proj, players, targets)
            print(f"  {name:<46}{e['pass_yds']:>9.2f}{e['rush_yds']:>9.2f}{e['receptions']:>7.3f}{e['rec_yds']:>9.2f}")
    conn.close()


if __name__ == "__main__":
    main()
