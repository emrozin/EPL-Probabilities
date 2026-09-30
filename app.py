"""Website: upcoming matches, recent results, league table, team pages, backtest and ratings.

Run locally:
    uvicorn app:app --reload
then open http://127.0.0.1:8000 in a browser.

build_site.py renders every page to static HTML for GitHub Pages. For that, links need
a path prefix (the repository name), which comes from the SITE_BASE environment variable.
"""

import math
import os
import re
import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.exceptions import HTTPException as StarletteHTTPException

from model import (TRAINING_YEARS, add_scores, fit_poisson, goal_markets, load_games, load_market,
                   load_totals, scoreline_grid)
from predict_upcoming import MODEL_NAME, MODEL_VERSION
from run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "data" / "sports.db"
BASE = os.environ.get("SITE_BASE", "").rstrip("/")  # "" locally, "/epl-probabilities" on GitHub Pages

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


def slugify(name: str) -> str:
    """'Nott'm Forest' -> 'nottm-forest'"""
    name = name.lower().replace("'", "").replace("&", "and")
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def team_url(name: str, season: int | None = None) -> str:
    return f"{BASE}/team/{slugify(name)}/" + (f"{season}/" if season is not None else "")


templates.env.globals["team_url"] = team_url
templates.env.globals["base"] = BASE


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    if exc.status_code == 404:
        return templates.TemplateResponse(request, "404.html", {"active": None}, status_code=404)
    return HTMLResponse(str(exc.detail), status_code=exc.status_code)


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


def load_goals(conn, model_name: str, model_version: str) -> pd.DataFrame:
    """One row per game: the prediction's expected goals (exp_home, exp_away)."""
    return pd.read_sql("""
        SELECT game_id AS id, exp_home, exp_away FROM prediction_goals
        WHERE model_name = ? AND model_version = ?
    """, conn, params=(model_name, model_version))


def load_best_predictions(conn) -> pd.DataFrame:
    """A prediction saved before kickoff where one exists, otherwise the backtest's."""
    live = load_predictions(conn, MODEL_NAME, MODEL_VERSION).merge(
        load_goals(conn, MODEL_NAME, MODEL_VERSION), on="id", how="left")
    back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION).merge(
        load_goals(conn, BACKTEST_NAME, BACKTEST_VERSION), on="id", how="left")
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


_cache: dict = {}


def data() -> dict:
    """Everything the pages read, loaded once and reused until the database file changes."""
    stamp = DB_PATH.stat().st_mtime
    if _cache.get("stamp") != stamp:
        with connect() as conn:
            _cache.clear()
            _cache.update(
                stamp=stamp,
                finished=load_finished(conn),
                predictions=load_best_predictions(conn),
                backtest=load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION),
                backtest_goals=load_goals(conn, BACKTEST_NAME, BACKTEST_VERSION),
                closing=load_market(conn, closing=True),
                totals=load_totals(conn, closing=True),
                games=load_games(conn),
            )
    return _cache


def cached_ratings() -> dict:
    """Current model ratings, refitted when the data or the date changes."""
    d, today = data(), date.today().isoformat()
    if d.get("ratings_day") != today:
        d["ratings"] = current_ratings(d["games"], today)
        d["ratings_day"] = today
    return d["ratings"]


def season_options(seasons, selected, url) -> list[dict]:
    """For the season picker: newest first, each with its page address."""
    return [{"label": season_label(s), "url": url(s), "selected": s == selected}
            for s in sorted(seasons, reverse=True)]


def pct(p: float) -> str:
    """0.083 -> '8.3', 0.42 -> '42': one decimal for small chances, where it matters."""
    return f"{p * 100:.1f}" if p < 0.095 else f"{round(p * 100)}"


def goals_info(exp_home, exp_away, market_over=None, score=None) -> dict:
    """Over/under 2.5, the likeliest scores, and (for a finished match) the chance of its exact score."""
    info = {"over": None, "market_over": None, "top_scores": [], "exact_pct": None}
    if exp_home is not None and pd.notna(exp_home) and pd.notna(exp_away):
        grid = scoreline_grid(exp_home, exp_away)
        markets = goal_markets(grid, top=3)
        info["over"] = round(markets["over_2.5"] * 100)
        info["top_scores"] = [{"score": f"{i}\u2013{j}", "pct": pct(prob)} for i, j, prob in markets["top_scores"]]
        if score is not None and max(score) < grid.shape[0]:
            info["exact_pct"] = pct(grid[score])
    if market_over is not None and pd.notna(market_over):
        info["market_over"] = round(market_over * 100)
    return info


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
    card["total_goals"] = int(m.home_score + m.away_score)
    card.update(goals_info(getattr(m, "exp_home", None) if has_prediction else None,
                           getattr(m, "exp_away", None), getattr(m, "mkt_over", None),
                           (int(m.home_score), int(m.away_score))))
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
# Every address ends with "/" so each page can be saved as <address>/index.html for static hosting.

@app.get("/", response_class=HTMLResponse)
def upcoming(request: Request):
    today = date.today().isoformat()
    with connect() as conn:
        matches = pd.read_sql("""
            SELECT g.id, g.date, h.name AS home, a.name AS away,
                   MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
                   MAX(CASE WHEN p.outcome = 'draw' THEN p.probability END) AS model_draw,
                   MAX(CASE WHEN p.outcome = 'away' THEN p.probability END) AS model_away,
                   MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
            FROM games g
            JOIN leagues l     ON l.id = g.league_id
            JOIN teams h       ON h.id = g.home_team_id
            JOIN teams a       ON a.id = g.away_team_id
            JOIN predictions p ON p.game_id = g.id
                              AND p.model_name = ? AND p.model_version = ?
            LEFT JOIN prediction_goals pg ON pg.game_id = g.id
                              AND pg.model_name = p.model_name AND pg.model_version = p.model_version
            WHERE l.code = 'EPL' AND g.status = 'scheduled' AND g.date >= ?
            GROUP BY g.id
            ORDER BY g.date, h.name
        """, conn, params=(MODEL_NAME, MODEL_VERSION, today))
        market = load_market(conn, closing=False)
        totals = load_totals(conn, closing=False)

    matches = matches.merge(market, on="id", how="left").merge(totals, on="id", how="left")
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
                **goals_info(m.exp_home, m.exp_away, m.mkt_over),
            })
        days.append({"label": day_label(match_date), "matches": day_matches})

    return templates.TemplateResponse(request, "upcoming.html", {"active": "upcoming", "days": days})


def week_of(dates: pd.Series) -> pd.Series:
    """The Monday starting each date's week, as 'YYYY-MM-DD'."""
    return pd.to_datetime(dates).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")


@app.get("/results/", response_class=HTMLResponse)
def results_latest(request: Request):
    return results_page(request, None)


@app.get("/results/{week}/", response_class=HTMLResponse)
def results_week(request: Request, week: str):
    return results_page(request, week)


def results_page(request: Request, week: str | None):
    d = data()
    df = d["finished"].merge(d["predictions"], on="id")
    if df.empty:
        return templates.TemplateResponse(request, "results.html", {"active": "results", "days": []})

    df["week"] = week_of(df["date"])
    weeks = sorted(df["week"].unique())
    if week is None:
        week = weeks[-1]
    elif week not in weeks:
        raise HTTPException(status_code=404)
    position = weeks.index(week)

    wk = (df[df["week"] == week].merge(d["closing"], on="id", how="left")
                                 .merge(d["totals"], on="id", how="left").sort_values(["date", "home"]))
    days = [{"label": day_label(day), "matches": [match_card(m) for m in group.itertuples()]}
            for day, group in wk.groupby("date", sort=True)]

    first, last = wk["date"].min(), wk["date"].max()
    week_url = lambda w: f"{BASE}/results/{w}/"
    return templates.TemplateResponse(request, "results.html", {
        "active": "results",
        "days": days,
        "summary": forecast_summary(wk),
        "week_label": day_label(first) if first == last else f"{day_label(first)} to {day_label(last)}",
        "previous_url": week_url(weeks[position - 1]) if position > 0 else None,
        "next_url": week_url(weeks[position + 1]) if position < len(weeks) - 1 else None,
    })


@app.get("/table/", response_class=HTMLResponse)
def table_latest(request: Request):
    return table_page(request, None)


@app.get("/table/{season}/", response_class=HTMLResponse)
def table_season(request: Request, season: int):
    return table_page(request, season)


def table_page(request: Request, season: int | None):
    finished = data()["finished"]
    if finished.empty:
        return templates.TemplateResponse(request, "table.html", {"active": "table", "rows": []})

    seasons = sorted(finished["season"].unique())
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    table = league_table(finished, season)
    relegation_from = len(table) - 2  # the bottom three go down

    rows = [{
        "position": r.position,
        "team": r.team,
        # Team pages cover seasons with model predictions; older tables link to the team's main page.
        "url": team_url(r.team, season if season >= START_SEASON else None),
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
        "is_current": season == seasons[-1],
        "has_xg": bool(table["xg_games"].sum()),
        "seasons": season_options(seasons, season, lambda s: f"{BASE}/table/{s}/"),
    })


@app.get("/team/{slug}/", response_class=HTMLResponse)
def team_latest(request: Request, slug: str):
    return team_page(request, slug, None)


@app.get("/team/{slug}/all/", response_class=HTMLResponse)
def team_all(request: Request, slug: str):
    return team_page(request, slug, "all")


@app.get("/team/{slug}/{season}/", response_class=HTMLResponse)
def team_season(request: Request, slug: str, season: int):
    return team_page(request, slug, season)


def team_page(request: Request, slug: str, season: int | str | None):
    """One season of a team's matches, or every season (season="all")."""
    d = data()
    finished = d["finished"]
    names = {slugify(n): n for n in pd.concat([finished["home"], finished["away"]]).unique()}
    if slug not in names:
        raise HTTPException(status_code=404)
    name = names[slug]

    played = finished[(finished["home"] == name) | (finished["away"] == name)]
    # Seasons with model predictions; teams only seen before then still get their seasons listed.
    seasons = sorted(s for s in played["season"].unique() if s >= START_SEASON) \
        or sorted(played["season"].unique())
    show_all = season == "all"
    if season is None:
        season = seasons[-1]
    elif not show_all and season not in seasons:
        raise HTTPException(status_code=404)
    latest = finished["season"].max()
    is_current = show_all or season == latest

    shown = played[played["season"].isin(seasons if show_all else [season])]
    matches = (shown.merge(d["predictions"], on="id", how="left")
                    .merge(d["closing"], on="id", how="left")
                    .merge(d["totals"], on="id", how="left")
                    .sort_values("date", ascending=False))

    # Group the matches by season, newest first (a single group when showing one season).
    groups = []
    for group_season, group in matches.groupby("season", sort=False):
        cards = []
        for m in group.itertuples():
            card = match_card(m)
            at_home = m.home == name
            card["venue"] = "Home" if at_home else "Away"
            card["outcome"] = "D" if m.result == "draw" else ("W" if (m.result == "home") == at_home else "L")
            cards.append(card)
        groups.append({"label": season_label(group_season), "matches": cards})

    if show_all:
        outcomes = [c["outcome"] for g in groups for c in g["matches"]]
        record = {"played": len(outcomes), "won": outcomes.count("W"),
                  "drawn": outcomes.count("D"), "lost": outcomes.count("L")}
        position = None
    else:
        standing = league_table(finished, season).set_index("team").loc[name]
        record = {"played": int(standing["played"]), "won": int(standing["won"]),
                  "drawn": int(standing["drawn"]), "lost": int(standing["lost"]),
                  "points": int(standing["points"]), "gd": signed_number(standing["gd"])}
        position = ordinal(int(standing["position"]))

    ratings = None
    if is_current and name in set(d["games"].loc[d["games"]["season"] == latest, ["home", "away"]].stack()):
        model = cached_ratings()
        if name in model["ratings"].index:
            r = model["ratings"].loc[name]
            ratings = rating_text(r["attack"], r["defense"])

    options = season_options(seasons, season, lambda s: team_url(name, s))
    if len(seasons) > 1:
        options.insert(0, {"label": "All seasons", "url": team_url(name) + "all/", "selected": show_all})

    return templates.TemplateResponse(request, "team.html", {
        "active": None,
        "team": name,
        "show_all": show_all,
        "season": None if show_all else season_label(season),
        "first_season": season_label(seasons[0]),
        "table_url": f"{BASE}/table/{latest if show_all else season}/",
        "is_current": is_current,
        "position": position,
        "record": record,
        "ratings": ratings,
        "summary": forecast_summary(matches),
        "groups": groups,
        "seasons": options,
    })


@app.get("/backtest/", response_class=HTMLResponse)
def backtest_page(request: Request):
    d = data()
    df = d["finished"].merge(d["backtest"], on="id").merge(d["closing"], on="id")
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
        "goals": goals_backtest(d),
        "development": [{"label": label, "value": f"{value:.4f}"} for label, value in DEVELOPMENT],
        "tuned_through": season_label(HELD_OUT_FROM - 1),
    })


def goals_backtest(d: dict) -> dict | None:
    """Over/under 2.5 and exact scores on the held-out seasons: model vs the closing market."""
    df = d["finished"].merge(d["backtest_goals"], on="id")
    df = df[df["season"] >= HELD_OUT_FROM].merge(d["totals"], on="id", how="left")
    if df.empty:
        return None
    over_p, exact_p, likeliest_hit = [], [], []
    for r in df.itertuples():
        grid = scoreline_grid(r.exp_home, r.exp_away)
        markets = goal_markets(grid, top=1)
        over_p.append(markets["over_2.5"])
        last = grid.shape[0] - 1
        exact_p.append(grid[min(r.home_score, last), min(r.away_score, last)])
        i, j, _ = markets["top_scores"][0]
        likeliest_hit.append(i == r.home_score and j == r.away_score)
    df["model_over"] = over_p
    went_over = (df["home_score"] + df["away_score"]) > 2.5

    def typical(p_over, mask):
        p_actual = np.where(went_over[mask], p_over[mask], 1 - p_over[mask])
        return typical_probability(-np.log(p_actual).mean())

    compared = df["mkt_over"].notna()
    return {
        "games": len(df),
        "compared": int(compared.sum()),
        "model_typical": typical(df["model_over"], compared) if compared.any() else None,
        "market_typical": typical(df["mkt_over"], compared) if compared.any() else None,
        "over_rate": round(went_over.mean() * 100),
        "exact_typical": typical_probability(-np.log(np.array(exact_p)).mean()),
        "likeliest_hit": round(np.mean(likeliest_hit) * 100),
        "from": season_label(HELD_OUT_FROM),
    }


@app.get("/ratings/", response_class=HTMLResponse)
def ratings(request: Request):
    games = data()["games"]
    model = cached_ratings()

    # Only show teams in the current season.
    current = games[games["season"] == games["season"].max()]
    teams = sorted(set(current["home"]) | set(current["away"]))
    r = model["ratings"].loc[[t for t in teams if t in model["ratings"].index]].copy()
    r["strength"] = np.log(r["attack"]) + np.log(r["defense"])
    r = r.sort_values("strength", ascending=False)

    rows = [{"team": team, **rating_text(row.attack, row.defense)} for team, row in r.iterrows()]
    return templates.TemplateResponse(request, "ratings.html", {
        "active": "ratings",
        "rows": rows,
        "as_of": day_label(date.today().isoformat()),
        "home_advantage": round((model["home_adv"] - 1) * 100),
        "matches_used": model["matches_used"],
    })
