"""Website: upcoming matches, recent results, league table, team pages, backtest and ratings.

Run locally:
    uvicorn app:app --reload
then open http://127.0.0.1:8000 in a browser.
"""

import math
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from model import TRAINING_YEARS, add_scores, fit_poisson, load_games, load_market
from predict_upcoming import MODEL_NAME, MODEL_VERSION
from run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON

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


def team_url(name: str) -> str:
    """'Nott'm Forest' -> '/team/Nott%27m%20Forest'"""
    return "/team/" + quote(name, safe="")


templates.env.globals["team_url"] = team_url


# ---------------------------------------------------------------- helpers

def connect():
    return closing(sqlite3.connect(DB_PATH))


def day_label(iso_date: str, with_year: bool = False) -> str:
    """'2026-10-03' -> 'Saturday 3 October' (or 'Sat 3 Oct 2026' with_year)"""
    d = pd.Timestamp(iso_date)
    if with_year:
        return f"{d:%a} {d.day} {d:%b %Y}"
    return f"{d:%A} {d.day} {d:%B}"


def season_label(season: int) -> str:
    """2024 -> '2024/25'"""
    return f"{season}/{(season + 1) % 100:02d}"


def ordinal(n: int) -> str:
    """1 -> '1st', 12 -> '12th', 22 -> '22nd'"""
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


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


def signed_number(value: float, decimals: int = 0) -> str:
    """Goal difference style: +7, −3, 0"""
    text = f"{abs(value):.{decimals}f}"
    if round(value, decimals) > 0:
        return "+" + text
    if round(value, decimals) < 0:
        return "\u2212" + text
    return text


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


def load_best_predictions(conn) -> pd.DataFrame:
    """A prediction saved before kickoff where one exists, otherwise the backtest's."""
    live = load_predictions(conn, MODEL_NAME, MODEL_VERSION)
    back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION)
    live["source"] = "Saved before kickoff"
    back["source"] = "Backtest"
    return pd.concat([live, back[~back["id"].isin(live["id"])]], ignore_index=True)


def load_finished(conn) -> pd.DataFrame:
    """Finished EPL matches with results and (where available) Understat xG."""
    df = pd.read_sql("""
        SELECT g.id, g.season, g.date, h.name AS home, a.name AS away, g.home_score, g.away_score,
               x.home_xg, x.away_xg
        FROM games g
        JOIN leagues l ON l.id = g.league_id
        JOIN teams h   ON h.id = g.home_team_id
        JOIN teams a   ON a.id = g.away_team_id
        LEFT JOIN game_xg x ON x.game_id = g.id AND x.source = 'understat'
        WHERE l.code = 'EPL' AND g.status = 'final'
    """, conn)
    df["result"] = np.select(
        [df["home_score"] > df["away_score"], df["home_score"] == df["away_score"]],
        ["home", "draw"], default="away")
    return df


def match_card(m) -> dict:
    """Everything a finished-match row needs (m has game, prediction and market columns)."""
    has_prediction = pd.notna(m.model_home)
    has_market = pd.notna(m.mkt_home)
    card = {
        "home": m.home,
        "away": m.away,
        "date": day_label(m.date, with_year=True),
        "score": f"{m.home_score}\u2013{m.away_score}",
        "result": m.result,
        "model": None, "market": None, "model_actual": None, "market_actual": None, "source": None,
    }
    if has_prediction:
        card["model"] = percentages(m.model_home, m.model_draw, m.model_away)
        card["model_actual"] = round(getattr(m, f"model_{m.result}") * 100)
        card["source"] = m.source
    if has_market:
        card["market"] = percentages(m.mkt_home, m.mkt_draw, m.mkt_away)
        card["market_actual"] = round(getattr(m, f"mkt_{m.result}") * 100)
    return card


def forecast_summary(df: pd.DataFrame) -> dict | None:
    """Typical probability for the actual result: model vs market, on matches that have both."""
    both = df.dropna(subset=["model_home", "mkt_home"])
    if both.empty:
        return None
    model_p = np.array([getattr(r, f"model_{r.result}") for r in both.itertuples()])
    market_p = np.array([getattr(r, f"mkt_{r.result}") for r in both.itertuples()])
    return {
        "games": len(both),
        "model_typical": typical_probability(-np.log(model_p).mean()),
        "market_typical": typical_probability(-np.log(market_p).mean()),
        "model_closer": int((model_p > market_p).sum()),
    }


def league_table(finished: pd.DataFrame, season: int) -> pd.DataFrame:
    """Standings for one season: sorted by points, goal difference, then goals scored."""
    g = finished[finished["season"] == season]
    home = pd.DataFrame({"team": g["home"], "date": g["date"], "gf": g["home_score"], "ga": g["away_score"],
                         "xgf": g["home_xg"], "xga": g["away_xg"]})
    away = pd.DataFrame({"team": g["away"], "date": g["date"], "gf": g["away_score"], "ga": g["home_score"],
                         "xgf": g["away_xg"], "xga": g["home_xg"]})
    rows = pd.concat([home, away], ignore_index=True)
    rows["outcome"] = np.select([rows["gf"] > rows["ga"], rows["gf"] == rows["ga"]], ["W", "D"], default="L")
    rows["points"] = rows["outcome"].map({"W": 3, "D": 1, "L": 0})

    table = rows.groupby("team").agg(
        played=("outcome", "size"),
        won=("outcome", lambda s: (s == "W").sum()),
        drawn=("outcome", lambda s: (s == "D").sum()),
        lost=("outcome", lambda s: (s == "L").sum()),
        gf=("gf", "sum"), ga=("ga", "sum"), points=("points", "sum"),
        xgf=("xgf", "sum"), xga=("xga", "sum"), xg_games=("xgf", "count"),
    )
    table["gd"] = table["gf"] - table["ga"]
    table["form"] = rows.sort_values("date").groupby("team")["outcome"].apply(lambda s: list(s.tail(5)))
    table = table.reset_index().sort_values(["points", "gd", "gf", "team"],
                                            ascending=[False, False, False, True]).reset_index(drop=True)
    table["position"] = table.index + 1
    return table


def current_ratings(games: pd.DataFrame, today: str) -> dict:
    """Fit the model on everything up to today, exactly as predictions are made."""
    window_start = (pd.Timestamp(today) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
    train = games[(games["date"] >= window_start) & (games["date"] < today)]
    model = fit_poisson(train, today)
    model["matches_used"] = len(train)
    return model


def rating_text(attack: float, defense: float) -> dict:
    scores = round((attack - 1) * 100)          # vs an average team
    concedes = round((1 / defense - 1) * 100)   # negative is good
    return {
        "scores": signed(scores),
        "scores_class": "good" if scores > 0 else "bad" if scores < 0 else "",
        "concedes": signed(concedes),
        "concedes_class": "good" if concedes < 0 else "bad" if concedes > 0 else "",
    }


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
        predictions = load_best_predictions(conn)
        market = load_market(conn, closing=True)

    df = finished.merge(predictions, on="id")
    if df.empty:
        return templates.TemplateResponse(request, "results.html", {"active": "results", "days": []})

    df["week"] = pd.to_datetime(df["date"]).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    weeks = sorted(df["week"].unique())
    current = week if week in weeks else weeks[-1]
    position = weeks.index(current)

    wk = df[df["week"] == current].merge(market, on="id", how="left").sort_values(["date", "home"])
    days = [{"label": day_label(d), "matches": [match_card(m) for m in group.itertuples()]}
            for d, group in wk.groupby("date", sort=True)]

    first, last = wk["date"].min(), wk["date"].max()
    return templates.TemplateResponse(request, "results.html", {
        "active": "results",
        "days": days,
        "summary": forecast_summary(wk),
        "games_in_week": len(wk),
        "week_label": day_label(first) if first == last else f"{day_label(first)} to {day_label(last)}",
        "previous_week": weeks[position - 1] if position > 0 else None,
        "next_week": weeks[position + 1] if position < len(weeks) - 1 else None,
    })


@app.get("/table", response_class=HTMLResponse)
def table_page(request: Request, season: int | None = None):
    with connect() as conn:
        finished = load_finished(conn)
    if finished.empty:
        return templates.TemplateResponse(request, "table.html", {"active": "table", "rows": []})

    seasons = sorted(finished["season"].unique(), reverse=True)
    season = season if season in seasons else seasons[0]
    table = league_table(finished, season)
    relegation_from = len(table) - 2  # the bottom three go down

    rows = [{
        "position": r.position,
        "team": r.team,
        "played": r.played, "won": r.won, "drawn": r.drawn, "lost": r.lost,
        "gf": r.gf, "ga": r.ga, "gd": signed_number(r.gd), "points": r.points,
        "xgf": f"{r.xgf:.1f}" if r.xg_games else "\u2013",
        "xga": f"{r.xga:.1f}" if r.xg_games else "\u2013",
        "form": r.form,
        "relegation": r.position >= relegation_from,
        "first_relegation": r.position == relegation_from,
    } for r in table.itertuples()]

    return templates.TemplateResponse(request, "table.html", {
        "active": "table",
        "rows": rows,
        "season": season_label(season),
        "is_current": season == seasons[0],
        "has_xg": bool(table["xg_games"].sum()),
        "seasons": [{"value": s, "label": season_label(s)} for s in seasons if s >= START_SEASON],
        "selected": season,
    })


@app.get("/team/{name}", response_class=HTMLResponse)
def team_page(request: Request, name: str, season: int | None = None):
    today = date.today().isoformat()
    with connect() as conn:
        finished = load_finished(conn)
        predictions = load_best_predictions(conn)
        market = load_market(conn, closing=True)
        games = load_games(conn)

    played = finished[(finished["home"] == name) | (finished["away"] == name)]
    if played.empty:
        raise HTTPException(status_code=404, detail=f"No Premier League matches found for {name}")

    seasons = sorted((s for s in played["season"].unique() if s >= START_SEASON), reverse=True) \
        or sorted(played["season"].unique(), reverse=True)
    season = season if season in seasons else seasons[0]
    is_current = season == finished["season"].max()

    matches = (played[played["season"] == season]
               .merge(predictions, on="id", how="left")
               .merge(market, on="id", how="left")
               .sort_values("date", ascending=False))
    cards = []
    for m in matches.itertuples():
        card = match_card(m)
        at_home = m.home == name
        card["opponent"] = m.away if at_home else m.home
        card["venue"] = "Home" if at_home else "Away"
        card["outcome"] = "D" if m.result == "draw" else ("W" if (m.result == "home") == at_home else "L")
        cards.append(card)

    table = league_table(finished, season)
    standing = table[table["team"] == name].iloc[0]

    ratings = None
    if is_current:
        model = current_ratings(games, today)
        if name in model["ratings"].index:
            r = model["ratings"].loc[name]
            ratings = rating_text(r["attack"], r["defense"])

    return templates.TemplateResponse(request, "team.html", {
        "active": None,
        "team": name,
        "season": season_label(season),
        "is_current": is_current,
        "position": ordinal(int(standing["position"])),
        "record": {"played": int(standing["played"]), "won": int(standing["won"]), "drawn": int(standing["drawn"]),
                   "lost": int(standing["lost"]), "points": int(standing["points"]), "gd": signed_number(standing["gd"])},
        "ratings": ratings,
        "summary": forecast_summary(matches),
        "matches": cards,
        "seasons": [{"value": s, "label": season_label(s)} for s in seasons],
        "selected": season,
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
    model = current_ratings(games, today)

    # Only show teams in the current season.
    current = games[games["season"] == games["season"].max()]
    teams = sorted(set(current["home"]) | set(current["away"]))
    r = model["ratings"].loc[teams].copy()
    r["strength"] = np.log(r["attack"]) + np.log(r["defense"])
    r = r.sort_values("strength", ascending=False)

    rows = [{"team": team, **rating_text(row.attack, row.defense)} for team, row in r.iterrows()]
    return templates.TemplateResponse(request, "ratings.html", {
        "active": "ratings",
        "rows": rows,
        "as_of": day_label(today),
        "home_advantage": round((model["home_adv"] - 1) * 100),
        "matches_used": model["matches_used"],
    })
