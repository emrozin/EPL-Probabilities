"""Website: upcoming match probabilities, recent results, backtest results and team ratings.

Run locally:
    uvicorn app:app --reload
then open http://127.0.0.1:8000 in a browser.
"""

import math
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model import TRAINING_YEARS, add_scores, fit_poisson, load_games, load_market
from predict_upcoming import MODEL_NAME, MODEL_VERSION
from run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "sports.db"

# How each model version scored on the held-out seasons while it was being developed
# (from poisson_model.ipynb, measured on 1,921 matches in September 2026).
DEVELOPMENT = [
    ("Goals only, initial settings", 0.9824),
    ("Goals only, tuned", 0.9813),
    ("Adding expected goals (xG)", 0.9778),
    ("xG with re-tuned recency weighting (current)", 0.9754),
    ("Closing betting market", 0.9592),
]

app = FastAPI(title="Premier League match probabilities")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


# ---------------------------------------------------------------- helpers

def connect():
    return closing(sqlite3.connect(DB_PATH))


def day_label(iso_date: str) -> str:
    """'2026-10-03' -> 'Saturday 3 October'"""
    d = pd.Timestamp(iso_date)
    return f"{d:%A} {d.day} {d:%B}"


def season_label(season: int) -> str:
    """2024 -> '2024/25'"""
    return f"{season}/{(season + 1) % 100:02d}"


def percentages(home: float, draw: float, away: float) -> dict:
    """Whole-number percentages that always add up to exactly 100."""
    raw = np.array([home, draw, away]) * 100
    rounded = np.floor(raw).astype(int)
    for i in np.argsort(raw - rounded)[::-1][: 100 - rounded.sum()]:
        rounded[i] += 1
    return dict(zip(["home", "draw", "away"], rounded.tolist()))


def signed(value: int) -> str:
    """12 -> '+12%', -5 -> '−5%' (a true minus sign), 0 -> '0%'"""
    if value > 0:
        return f"+{value}%"
    if value < 0:
        return f"\u2212{-value}%"
    return "0%"


def typical_probability(log_loss: float) -> str:
    """Log loss as a percentage: the typical probability given to what actually happened.

    One decimal place, because the model and the market are often less than a point apart.
    """
    return f"{math.exp(-log_loss) * 100:.1f}"


def load_predictions(conn, model_name: str, model_version: str) -> pd.DataFrame:
    """One row per game: model_home, model_draw, model_away."""
    return pd.read_sql("""
        SELECT game_id AS id,
               MAX(CASE WHEN outcome = 'home' THEN probability END) AS model_home,
               MAX(CASE WHEN outcome = 'draw' THEN probability END) AS model_draw,
               MAX(CASE WHEN outcome = 'away' THEN probability END) AS model_away
        FROM predictions
        WHERE model_name = ? AND model_version = ?
        GROUP BY game_id
    """, conn, params=(model_name, model_version))


def load_finished(conn) -> pd.DataFrame:
    df = pd.read_sql("""
        SELECT g.id, g.season, g.date, h.name AS home, a.name AS away, g.home_score, g.away_score
        FROM games g
        JOIN leagues l ON l.id = g.league_id
        JOIN teams h   ON h.id = g.home_team_id
        JOIN teams a   ON a.id = g.away_team_id
        WHERE l.code = 'EPL' AND g.status = 'final'
    """, conn)
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        ["home", "draw"], default="away")
    return df


# ---------------------------------------------------------------- pages

@app.get("/", response_class=HTMLResponse)
def upcoming(request: Request):
    today = date.today().isoformat()
    with connect() as conn:
        matches = pd.read_sql("""
            SELECT g.id, g.date, h.name AS home, a.name AS away,
                   MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
                   MAX(CASE WHEN p.outcome = 'draw' THEN p.probability END) AS model_draw,
                   MAX(CASE WHEN p.outcome = 'away' THEN p.probability END) AS model_away
            FROM games g
            JOIN leagues l     ON l.id = g.league_id
            JOIN teams h       ON h.id = g.home_team_id
            JOIN teams a       ON a.id = g.away_team_id
            JOIN predictions p ON p.game_id = g.id
                              AND p.model_name = ? AND p.model_version = ?
            WHERE l.code = 'EPL' AND g.status = 'scheduled' AND g.date >= ?
            GROUP BY g.id
            ORDER BY g.date, h.name
        """, conn, params=(MODEL_NAME, MODEL_VERSION, today))
        market = load_market(conn, closing=False)

    matches = matches.merge(market, on="id", how="left")
    days = []
    for match_date, group in matches.groupby("date", sort=True):
        day_matches = []
        for m in group.itertuples():
            has_market = pd.notna(m.mkt_home)
            day_matches.append({
                "home": m.home,
                "away": m.away,
                "model": percentages(m.model_home, m.model_draw, m.model_away),
                "market": percentages(m.mkt_home, m.mkt_draw, m.mkt_away) if has_market else None,
            })
        days.append({"label": day_label(match_date), "matches": day_matches})

    return templates.TemplateResponse(request, "upcoming.html", {"active": "upcoming", "days": days})


@app.get("/results", response_class=HTMLResponse)
def results(request: Request, week: str | None = None):
    with connect() as conn:
        finished = load_finished(conn)
        live = load_predictions(conn, MODEL_NAME, MODEL_VERSION)
        back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION)
        market = load_market(conn, closing=True)

    # Prefer a prediction saved before kickoff; otherwise use the backtest's.
    live["source"] = "Saved before kickoff"
    back["source"] = "Backtest"
    predictions = pd.concat([live, back[~back["id"].isin(live["id"])]])
    df = finished.merge(predictions, on="id")
    if df.empty:
        return templates.TemplateResponse(request, "results.html", {"active": "results", "days": []})

    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    weeks = sorted(df["week"].unique())
    current = week if week in weeks else weeks[-1]
    position = weeks.index(current)

    wk = df[df["week"] == current].merge(market, on="id", how="left").sort_values(["date", "home"])
    wk["model_actual"] = [row[f"model_{row['result']}"] for _, row in wk.iterrows()]
    wk["mkt_actual"] = [row[f"mkt_{row['result']}"] for _, row in wk.iterrows()]

    days = []
    for match_date, group in wk.groupby("date", sort=True):
        day_matches = []
        for m in group.itertuples():
            has_market = pd.notna(m.mkt_home)
            day_matches.append({
                "home": m.home,
                "away": m.away,
                "score": f"{m.home_score}\u2013{m.away_score}",
                "result": m.result,
                "model": percentages(m.model_home, m.model_draw, m.model_away),
                "market": percentages(m.mkt_home, m.mkt_draw, m.mkt_away) if has_market else None,
                "model_actual": round(m.model_actual * 100),
                "market_actual": round(m.mkt_actual * 100) if has_market else None,
                "source": m.source,
            })
        days.append({"label": day_label(match_date), "matches": day_matches})

    compared = wk.dropna(subset=["mkt_actual"])
    summary = {
        "games": len(wk),
        "model_typical": typical_probability(-np.log(wk["model_actual"]).mean()),
        "market_typical": typical_probability(-np.log(compared["mkt_actual"]).mean()) if len(compared) else None,
        "model_closer": int((compared["model_actual"] > compared["mkt_actual"]).sum()),
        "compared": len(compared),
    }
    first, last = wk["date"].min(), wk["date"].max()
    week_label = day_label(first) if first == last else f"{day_label(first)} to {day_label(last)}"

    return templates.TemplateResponse(request, "results.html", {
        "active": "results",
        "days": days,
        "summary": summary,
        "week_label": week_label,
        "previous_week": weeks[position - 1] if position > 0 else None,
        "next_week": weeks[position + 1] if position < len(weeks) - 1 else None,
    })


@app.get("/backtest", response_class=HTMLResponse)
def backtest_page(request: Request):
    with connect() as conn:
        finished = load_finished(conn)
        predictions = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION)
        market = load_market(conn, closing=True)

    df = finished.merge(predictions, on="id").merge(market, on="id")
    if df.empty:
        return templates.TemplateResponse(request, "backtest.html", {"active": "backtest", "seasons": []})
    add_scores(df, "model")
    add_scores(df, "mkt")

    by_season = (df.groupby("season")
                   .agg(games=("id", "size"), model=("model_logloss", "mean"), market=("mkt_logloss", "mean"))
                   .reset_index())
    by_season["gap"] = by_season["model"] - by_season["market"]
    widest = by_season["gap"].abs().max() or 1
    seasons = [{
        "label": season_label(r.season),
        "games": r.games,
        "model": f"{r.model:.3f}",
        "market": f"{r.market:.3f}",
        "gap": f"{abs(r.gap):.3f}",
        "market_better": r.gap > 0,
        "width": round(abs(r.gap) / widest * 100, 1),
        "held_out": r.season >= HELD_OUT_FROM,
    } for r in by_season.itertuples()]

    held = df[df["season"] >= HELD_OUT_FROM]
    headline = {
        "games": len(held),
        "from": season_label(HELD_OUT_FROM),
        "model": f"{held['model_logloss'].mean():.4f}",
        "market": f"{held['mkt_logloss'].mean():.4f}",
        "model_typical": typical_probability(held["model_logloss"].mean()),
        "market_typical": typical_probability(held["mkt_logloss"].mean()),
    }

    return templates.TemplateResponse(request, "backtest.html", {
        "active": "backtest",
        "seasons": seasons,
        "headline": headline,
        "development": [{"label": label, "value": f"{value:.4f}"} for label, value in DEVELOPMENT],
        "tuned_through": season_label(HELD_OUT_FROM - 1),
    })


@app.get("/ratings", response_class=HTMLResponse)
def ratings(request: Request):
    today = date.today().isoformat()
    with connect() as conn:
        games = load_games(conn)

    window_start = (pd.Timestamp(today) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
    train = games[(games["date"] >= window_start) & (games["date"] < today)]
    model = fit_poisson(train, today)

    # Only show teams in the current season.
    current = games[games["season"] == games["season"].max()]
    teams = sorted(set(current["home"]) | set(current["away"]))
    r = model["ratings"].loc[teams].copy()
    r["strength"] = np.log(r["attack"]) + np.log(r["defense"])
    r = r.sort_values("strength", ascending=False)

    rows = []
    for team, row in r.iterrows():
        scores = round((row.attack - 1) * 100)          # vs an average team
        concedes = round((1 / row.defense - 1) * 100)   # negative is good
        rows.append({
            "team": team,
            "scores": signed(scores),
            "scores_class": "good" if scores > 0 else "bad" if scores < 0 else "",
            "concedes": signed(concedes),
            "concedes_class": "good" if concedes < 0 else "bad" if concedes > 0 else "",
        })

    return templates.TemplateResponse(request, "ratings.html", {
        "active": "ratings",
        "rows": rows,
        "as_of": day_label(today),
        "home_advantage": round((model["home_adv"] - 1) * 100),
        "matches_used": len(train),
    })
