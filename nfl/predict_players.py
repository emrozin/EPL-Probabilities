"""Project players' stats for upcoming NFL games and save the projections.

Usage:
    python -m nfl.predict_players            # on your computer
    python -m nfl.predict_players --record   # the automated run: also updates the official record

Projects every game that has a live team prediction and hasn't kicked off (run after
nfl.predict_upcoming), using the team model's expected points. Like game predictions, a projection
can be refreshed until kickoff and never changes after it. The official record is
records/nfl_player_projections.csv, written only by the automated run.
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from db import DB_PATH, SCHEMA_PATH
from nfl.players import Projector, load, load_injuries, load_snaps
from nfl.predict_upcoming import EASTERN, MODEL_NAME, MODEL_VERSION, has_started

RECORD_PATH = Path("records/nfl_player_projections.csv")
STAT_COLUMNS = ["pass_att", "pass_yds", "pass_tds", "ints", "targets", "receptions", "rec_yds", "rec_tds",
                "carries", "rush_yds", "rush_tds", "anytime_td"]
COLUMNS = ["game", "date", "team", "player_id", "name", "position", *STAT_COLUMNS, "created_at"]


def restore(conn) -> int:
    if not RECORD_PATH.exists():
        return 0
    record = pd.read_csv(RECORD_PATH, dtype={"game": str, "player_id": str, "created_at": str},
                         float_precision="round_trip")
    ids = dict(conn.execute("SELECT source_game_id, game_id FROM nfl_game_info"))
    rows = [(ids[r.game], r.player_id, r.name, r.position, r.team, *[getattr(r, c) for c in STAT_COLUMNS], r.created_at)
            for r in record.itertuples() if r.game in ids]
    conn.executemany(f"INSERT OR REPLACE INTO nfl_player_projections (game_id, player_id, name, position, team, "
                     f"{', '.join(STAT_COLUMNS)}, created_at) VALUES ({', '.join('?' * (len(STAT_COLUMNS) + 6))})", rows)
    conn.commit()
    return len(rows)


def export(conn) -> None:
    current = pd.read_sql(f"""SELECT i.source_game_id AS game, g.date, p.team, p.player_id, p.name, p.position,
                                     {', '.join('p.' + c for c in STAT_COLUMNS)}, p.created_at
                              FROM nfl_player_projections p JOIN games g ON g.id = p.game_id
                              JOIN nfl_game_info i ON i.game_id = g.id""", conn)
    old = (pd.read_csv(RECORD_PATH, dtype={"game": str, "player_id": str, "created_at": str}, float_precision="round_trip")
           if RECORD_PATH.exists() else pd.DataFrame(columns=COLUMNS))
    old = old[~old["game"].isin(set(current["game"]))]   # a re-projected game replaces all its old rows
    combined = pd.concat([old.reindex(columns=COLUMNS), current], ignore_index=True).sort_values(
        ["date", "game", "team", "name"])
    RECORD_PATH.parent.mkdir(exist_ok=True)
    combined[COLUMNS].to_csv(RECORD_PATH, index=False)


def main(argv: list[str] | None = None, now: datetime | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    recording = "--record" in argv
    now = now or datetime.now(EASTERN)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_PATH.read_text())
    restored = restore(conn)
    recorded = {r[0] for r in conn.execute("SELECT DISTINCT game_id FROM nfl_player_projections")}

    upcoming = pd.read_sql("""
        SELECT g.id, g.date, g.kickoff, MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
        FROM games g JOIN prediction_goals pg ON pg.game_id = g.id AND pg.model_name = ? AND pg.model_version = ?
        WHERE g.status = 'scheduled' GROUP BY g.id""", conn, params=(MODEL_NAME, MODEL_VERSION))
    upcoming = upcoming[[not has_started(r.date, r.kickoff, now) for r in upcoming.itertuples()]]
    if not recording:
        upcoming = upcoming[~upcoming["id"].isin(recorded)]   # locally, the official record stands
    if restored:
        print(f"Loaded {restored} recorded player projections from {RECORD_PATH}.")
    if upcoming.empty:
        print("No upcoming NFL games to project players for.")
        if recording:
            export(conn)
        conn.close()
        return

    games, teams, players = load(conn)
    if players.empty:
        raise SystemExit("No player stats found. Run python -m nfl.import_nfl_stats first.")
    projections = Projector(games, teams, players, injuries=load_injuries(conn), snaps=load_snaps(conn)).run(
        {r.id: (r.exp_home, r.exp_away) for r in upcoming.itertuples()})
    with conn:
        conn.executemany("DELETE FROM nfl_player_projections WHERE game_id = ?", [(int(g),) for g in upcoming["id"]])
        for r in projections.itertuples():
            conn.execute(f"""INSERT INTO nfl_player_projections (game_id, player_id, name, position, team,
                                 {', '.join(STAT_COLUMNS)}) VALUES ({', '.join('?' * (len(STAT_COLUMNS) + 5))})""",
                         (int(r.game_id), r.player_id, r.name, r.position, r.team,
                          *[float(getattr(r, c, 0) or 0) for c in STAT_COLUMNS]))
    if recording:
        export(conn)
    conn.close()
    print(f"Saved projections for {len(projections)} players in {len(upcoming)} games.")


if __name__ == "__main__":
    main()
