"""Predict upcoming EPL fixtures and save the predictions.

Usage:
    python predict_upcoming.py

Fits the model on finished matches up to today, predicts every scheduled game
from today on, and stores the probabilities in the predictions table. Re-running
before kickoff replaces a game's prediction with a fresher one; games already
played are never touched, so stored predictions form a genuine pre-match track
record. Run it before the first kickoff of a round.
"""

import sqlite3
from datetime import date

import pandas as pd

from model import TRAINING_YEARS, fit_poisson, load_games, load_market, predict

DB_PATH = "data/sports.db"
MODEL_NAME = "poisson_xg"
MODEL_VERSION = "v2"  # 180-day half-life, shrinkage 0.5, xG weight 0.5


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    today = date.today().isoformat()

    upcoming = pd.read_sql("""
        SELECT g.id, g.date, h.name AS home, a.name AS away
        FROM games g
        JOIN leagues l ON l.id = g.league_id
        JOIN teams h   ON h.id = g.home_team_id
        JOIN teams a   ON a.id = g.away_team_id
        WHERE l.code = 'EPL' AND g.status = 'scheduled' AND g.date >= ?
        ORDER BY g.date, h.name
    """, conn, params=(today,))
    if upcoming.empty:
        print("No upcoming fixtures in the database. Run import_fixtures.py first "
              "(weekend fixtures are posted on Friday afternoons).")
        return

    # Train only on matches finished before today, exactly like the backtest.
    games = load_games(conn)
    window_start = (pd.Timestamp(today) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
    train = games[(games["date"] >= window_start) & (games["date"] < today)]
    model = fit_poisson(train, today)
    known = set(model["ratings"].index)

    market = load_market(conn, closing=False).set_index("id")

    rows = []
    for g in upcoming.itertuples():
        if g.home not in known or g.away not in known:
            print(f"Skipped {g.home} v {g.away}: no recent matches to rate a team on")
            continue
        p = predict(model, g.home, g.away)

        for outcome in ("home", "draw", "away"):
            conn.execute(
                """
                INSERT INTO predictions (game_id, model_name, model_version, outcome, probability)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (game_id, model_name, model_version, outcome)
                DO UPDATE SET probability = excluded.probability,
                              created_at  = datetime('now')
                """,
                (g.id, MODEL_NAME, MODEL_VERSION, outcome, float(p[outcome])),
            )

        row = {
            "date": g.date,
            "match": f"{g.home} v {g.away}",
            "exp goals": f"{p['exp_home']:.2f}-{p['exp_away']:.2f}",
            "model H/D/A": f"{p['home']:.0%} / {p['draw']:.0%} / {p['away']:.0%}",
        }
        if g.id in market.index:
            m = market.loc[g.id]
            row["market H/D/A"] = f"{m['mkt_home']:.0%} / {m['mkt_draw']:.0%} / {m['mkt_away']:.0%}"
            row["source"] = m["benchmark"]
        rows.append(row)

    conn.commit()
    conn.close()

    print(f"Model trained on {len(train)} matches up to {today}. Saved predictions for {len(rows)} games:\n")
    print(pd.DataFrame(rows).fillna("-").to_string(index=False))


if __name__ == "__main__":
    main()
