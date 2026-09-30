"""Website: upcoming match probabilities and team ratings.

Run locally:
    uvicorn app:app --reload
then open http://127.0.0.1:8000 in a browser.
"""

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

from model import TRAINING_YEARS, fit_poisson, load_games, load_market
from predict_upcoming import MODEL_NAME, MODEL_VERSION

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "sports.db"

app = FastAPI(title="Premier League match probabilities")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
templates = Jinja2Templates(directory=ROOT / "templates")


def day_label(iso_date: str) -> str:
    """'2026-10-03' -> 'Saturday 3 October'"""
    d = pd.Timestamp(iso_date)
    return f"{d:%A} {d.day} {d:%B}"


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


@app.get("/", response_class=HTMLResponse)
def upcoming(request: Request):
    today = date.today().isoformat()
    with closing(sqlite3.connect(DB_PATH)) as conn:
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
                "source": m.benchmark if has_market else None,
            })
        days.append({"label": day_label(match_date), "matches": day_matches})

    return templates.TemplateResponse(request, "upcoming.html", {"active": "upcoming", "days": days})


@app.get("/ratings", response_class=HTMLResponse)
def ratings(request: Request):
    today = date.today().isoformat()
    with closing(sqlite3.connect(DB_PATH)) as conn:
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
