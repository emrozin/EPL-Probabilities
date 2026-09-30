"""The official record of live predictions, kept in the repository as a CSV file.

The database isn't in the repository (it's rebuilt from the data sources), so live
predictions are saved here to persist between runs, and so every prediction, with the
time it was made, is visible in the Git history before the match is played.

Matches are identified by season, date and team names rather than database ids,
because ids change whenever the database is rebuilt.
"""

import sqlite3
from pathlib import Path

import pandas as pd

RECORD_PATH = Path("records/live_predictions.csv")
COLUMNS = ["season", "date", "home", "away", "model_name", "model_version",
           "home_win", "draw", "away_win", "exp_home_goals", "exp_away_goals", "created_at"]
KEY = ["season", "date", "home", "away", "model_name", "model_version"]


def read_record() -> pd.DataFrame:
    if not RECORD_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    # round_trip: read numbers back exactly as written (the default reader can differ in the last digit)
    return pd.read_csv(RECORD_PATH, dtype={"date": str, "created_at": str}, float_precision="round_trip")


def restore(conn: sqlite3.Connection) -> int:
    """Copy every recorded prediction into the database (replacing any local version). Returns how many."""
    record = read_record()
    restored = 0
    for r in record.itertuples():
        found = conn.execute("""
            SELECT g.id FROM games g
            JOIN leagues l ON l.id = g.league_id
            JOIN teams h   ON h.id = g.home_team_id
            JOIN teams a   ON a.id = g.away_team_id
            WHERE l.code = 'EPL' AND g.season = ? AND g.date = ? AND h.name = ? AND a.name = ?
        """, (int(r.season), r.date, r.home, r.away)).fetchone()
        if found is None:
            continue  # e.g. a fixture that was postponed and hasn't been re-imported
        created_at = r.created_at if isinstance(r.created_at, str) else pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        for outcome, probability in (("home", r.home_win), ("draw", r.draw), ("away", r.away_win)):
            conn.execute("""
                INSERT OR REPLACE INTO predictions
                    (game_id, model_name, model_version, outcome, probability, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (found[0], r.model_name, r.model_version, outcome, float(probability), created_at))
        exp_home, exp_away = getattr(r, "exp_home_goals", None), getattr(r, "exp_away_goals", None)
        if pd.notna(exp_home) and pd.notna(exp_away):  # older records don't have expected goals
            conn.execute("""
                INSERT OR REPLACE INTO prediction_goals
                    (game_id, model_name, model_version, exp_home, exp_away, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (found[0], r.model_name, r.model_version, float(exp_home), float(exp_away), created_at))
        restored += 1
    conn.commit()
    return restored


def export(conn: sqlite3.Connection, model_name: str, model_version: str) -> int:
    """Write the database's live predictions to the record, keeping any recorded rows it doesn't have."""
    current = pd.read_sql("""
        SELECT g.season, g.date, h.name AS home, a.name AS away, p.model_name, p.model_version,
               MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS home_win,
               MAX(CASE WHEN p.outcome = 'draw' THEN p.probability END) AS draw,
               MAX(CASE WHEN p.outcome = 'away' THEN p.probability END) AS away_win,
               MAX(pg.exp_home) AS exp_home_goals,
               MAX(pg.exp_away) AS exp_away_goals,
               MAX(p.created_at) AS created_at
        FROM predictions p
        LEFT JOIN prediction_goals pg ON pg.game_id = p.game_id
                                     AND pg.model_name = p.model_name AND pg.model_version = p.model_version
        JOIN games g ON g.id = p.game_id
        JOIN teams h ON h.id = g.home_team_id
        JOIN teams a ON a.id = g.away_team_id
        WHERE p.model_name = ? AND p.model_version = ?
        GROUP BY p.game_id
    """, conn, params=(model_name, model_version))
    combined = (pd.concat([read_record().reindex(columns=COLUMNS), current], ignore_index=True)
                  .drop_duplicates(subset=KEY, keep="last")          # the database's version wins
                  .sort_values(["date", "home"]))
    RECORD_PATH.parent.mkdir(exist_ok=True)
    # Probabilities are written at full precision, so restoring the record gives back exactly what was saved.
    combined[COLUMNS].to_csv(RECORD_PATH, index=False)
    return len(combined)
