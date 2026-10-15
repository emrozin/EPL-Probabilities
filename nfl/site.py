"""NFL pages of the website, under /nfl/: upcoming games, results, standings, team and game pages,
track record, backtest and team ratings. app.py includes them.

Games are listed the American way, away team first ("SEA at DEN"); in the probability bars
the away team is gold on the left and the home team blue on the right.
"""

import sqlite3
from contextlib import closing
from datetime import date, datetime

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from nfl.elo import load_nfl_games
from nfl.model import HALF_LIFE_DAYS, INEXPERIENCE_STARTS, MARGIN_SD, TRAINING_YEARS, add_qb_changes, fit, qb_value
from nfl.predict_upcoming import MODEL_NAME
from nfl.run_backtest import BACKTEST_NAME, BACKTEST_VERSION, HELD_OUT_FROM, START_SEASON
from web import BASE, ROOT, day_label, ordinal, signed_number, slugify, templates, typical_probability

DB_PATH = ROOT / "data" / "sports.db"
router = APIRouter(prefix="/nfl")

RECORD_URL = "https://github.com/emrozin/Sports-Probabilities/blob/main/records/nfl_live_predictions.csv"
RECORD_HISTORY_URL = "https://github.com/emrozin/Sports-Probabilities/commits/main/records/nfl_live_predictions.csv"
PLAYER_RECORD_URL = "https://github.com/emrozin/Sports-Probabilities/blob/main/records/nfl_player_projections.csv"

# Held-out results (2018-2025, 2,219 games, ties excluded) from nfl/tune_model.py, October 2026.
DEVELOPMENT = [
    ("Elo ratings (baseline)", 0.6382),
    ("Points model: offense and defense ratings", 0.6374),
    ("+ usual-starter quarterback flag", 0.6326),
    ("+ individual quarterback ratings, replacing the flag (current)", 0.6298),
    ("Betting market (moneylines)", 0.6089),
]

# Player projections vs simple benchmarks, held-out seasons 2018-2025 (nfl/players.py), October 2026:
# average miss per player-game.
PLAYER_ACCURACY = [
    ("Passing yards", "61.4", "71.6", "70.0"),
    ("Passing touchdowns", "0.92", "1.00", "0.98"),
    ("Rushing yards", "8.2", "8.9", "8.7"),
    ("Receptions", "1.28", "1.38", "1.34"),
    ("Receiving yards", "17.0", "18.6", "18.0"),
]

# Current names and divisions. Relocated teams appear under their current name throughout
# (Oakland Raiders -> Las Vegas, San Diego Chargers -> Los Angeles, St. Louis Rams -> Los Angeles).
TEAMS = {
    "BUF": ("Buffalo Bills", "AFC East"), "MIA": ("Miami Dolphins", "AFC East"),
    "NE": ("New England Patriots", "AFC East"), "NYJ": ("New York Jets", "AFC East"),
    "BAL": ("Baltimore Ravens", "AFC North"), "CIN": ("Cincinnati Bengals", "AFC North"),
    "CLE": ("Cleveland Browns", "AFC North"), "PIT": ("Pittsburgh Steelers", "AFC North"),
    "HOU": ("Houston Texans", "AFC South"), "IND": ("Indianapolis Colts", "AFC South"),
    "JAX": ("Jacksonville Jaguars", "AFC South"), "TEN": ("Tennessee Titans", "AFC South"),
    "DEN": ("Denver Broncos", "AFC West"), "KC": ("Kansas City Chiefs", "AFC West"),
    "LV": ("Las Vegas Raiders", "AFC West"), "LAC": ("Los Angeles Chargers", "AFC West"),
    "DAL": ("Dallas Cowboys", "NFC East"), "NYG": ("New York Giants", "NFC East"),
    "PHI": ("Philadelphia Eagles", "NFC East"), "WAS": ("Washington Commanders", "NFC East"),
    "CHI": ("Chicago Bears", "NFC North"), "DET": ("Detroit Lions", "NFC North"),
    "GB": ("Green Bay Packers", "NFC North"), "MIN": ("Minnesota Vikings", "NFC North"),
    "ATL": ("Atlanta Falcons", "NFC South"), "CAR": ("Carolina Panthers", "NFC South"),
    "NO": ("New Orleans Saints", "NFC South"), "TB": ("Tampa Bay Buccaneers", "NFC South"),
    "ARI": ("Arizona Cardinals", "NFC West"), "LA": ("Los Angeles Rams", "NFC West"),
    "SF": ("San Francisco 49ers", "NFC West"), "SEA": ("Seattle Seahawks", "NFC West"),
}
DIVISIONS_FROM = 2002   # the current eight divisions began in 2002
PLAYOFF_ROUNDS = {"WC": "Wild Card round", "DIV": "Divisional round", "CON": "Conference championships",
                  "SB": "Super Bowl"}


# ---------------------------------------------------------------- formatting

def team_name(code: str) -> str:
    return TEAMS.get(code, (code,))[0]


def nickname(code: str) -> str:
    """'KC' -> 'Chiefs': the last word of the name ('Commanders', '49ers')."""
    return team_name(code).split()[-1]


def team_url(code: str, season: int | None = None) -> str:
    return f"{BASE}/nfl/team/{slugify(team_name(code))}/" + (f"{season}/" if season is not None else "")


def game_url(source_id: str) -> str:
    """'2026_06_SEA_DEN' -> '/nfl/game/2026-06-sea-den/'"""
    return f"{BASE}/nfl/game/{source_id.lower().replace('_', '-')}/"


templates.env.globals["nfl_team_url"] = team_url
templates.env.globals["nfl_name"] = team_name


def week_label(game_type: str, week: int) -> str:
    return PLAYOFF_ROUNDS.get(game_type, f"Week {week}")


def kickoff_label(kickoff: str | None) -> str | None:
    """'20:15' -> '8:15 PM ET'"""
    if not kickoff or not isinstance(kickoff, str):
        return None
    t = datetime.strptime(kickoff, "%H:%M")
    return f"{t.hour % 12 or 12}:{t.minute:02d} {'AM' if t.hour < 12 else 'PM'} ET"


def two_way(p_home: float) -> dict:
    """Whole-number percentages for away and home that add up to 100."""
    home = int(round(p_home * 100))
    return {"away": 100 - home, "home": home}


def line_text(home_margin: float | None, home: str, away: str, decimals: int = 1) -> str | None:
    """Expected home margin -> 'KC by 3.5' (the favorite) or 'Even'."""
    if home_margin is None or pd.isna(home_margin):
        return None
    if round(abs(home_margin), decimals) == 0:
        return "Even"
    return f"{home if home_margin > 0 else away} by {abs(home_margin):.{decimals}f}"


def no_vig(home_moneyline, away_moneyline) -> float | None:
    """Market home win probability from American moneylines, with the bookmaker's margin removed."""
    if pd.isna(home_moneyline) or pd.isna(away_moneyline):
        return None
    implied = lambda ml: -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)
    h, a = implied(float(home_moneyline)), implied(float(away_moneyline))
    return h / (h + a)


def outcome_letter(team_score: int, opponent_score: int) -> str:
    return "W" if team_score > opponent_score else "L" if team_score < opponent_score else "T"


# ---------------------------------------------------------------- data

def connect():
    return closing(sqlite3.connect(DB_PATH))


def load_live(conn) -> pd.DataFrame:
    """Live predictions from every model version; for each game, the most recently saved one."""
    df = pd.read_sql("""
        SELECT p.game_id AS id, p.model_version, MAX(p.created_at) AS created_at,
               MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
               MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
        FROM predictions p
        LEFT JOIN prediction_goals pg ON pg.game_id = p.game_id
                                     AND pg.model_name = p.model_name AND pg.model_version = p.model_version
        WHERE p.model_name = ?
        GROUP BY p.game_id, p.model_version
    """, conn, params=(MODEL_NAME,))
    df = df.sort_values(["created_at", "model_version"]).drop_duplicates("id", keep="last")
    return df[["id", "model_home", "exp_home", "exp_away"]]


def load_predictions(conn, model_name: str, model_version: str) -> pd.DataFrame:
    return pd.read_sql("""
        SELECT p.game_id AS id, MAX(CASE WHEN p.outcome = 'home' THEN p.probability END) AS model_home,
               MAX(pg.exp_home) AS exp_home, MAX(pg.exp_away) AS exp_away
        FROM predictions p
        LEFT JOIN prediction_goals pg ON pg.game_id = p.game_id
                                     AND pg.model_name = p.model_name AND pg.model_version = p.model_version
        WHERE p.model_name = ? AND p.model_version = ?
        GROUP BY p.game_id
    """, conn, params=(model_name, model_version))


def load_closing(conn) -> pd.DataFrame:
    """Closing market home win probability (moneylines, margin removed) for finished games."""
    df = pd.read_sql("""
        SELECT h.game_id AS id, 1 / h.price AS h, 1 / a.price AS a
        FROM odds h JOIN odds a ON a.game_id = h.game_id AND a.market = 'moneyline' AND a.outcome = 'away'
        WHERE h.market = 'moneyline' AND h.outcome = 'home' AND h.bookmaker = 'nflverse'
    """, conn)
    df["mkt_home"] = df["h"] / (df["h"] + df["a"])
    return df[["id", "mkt_home"]]


_cache: dict = {}


GAME_COLUMNS = ["id", "season", "date", "status", "home", "away", "home_score", "away_score", "neutral_site",
                "game_type", "week", "spread_line", "home_qb", "away_qb", "home_qb_change", "away_qb_change",
                "home_qb_expected", "away_qb_expected", "home_qb_starts", "away_qb_starts",
                "kickoff", "source_game_id", "total_line", "home_moneyline", "away_moneyline", "stadium", "overtime"]


def empty_data(stamp) -> dict:
    """No NFL data yet (a database without NFL games): every page shows its empty state."""
    prediction_columns = ["id", "model_home", "exp_home", "exp_away", "source"]
    return {"stamp": stamp, "games": pd.DataFrame(columns=GAME_COLUMNS),
            "predictions": pd.DataFrame(columns=prediction_columns),
            "backtest": pd.DataFrame(columns=prediction_columns), "closing": pd.DataFrame(columns=["id", "mkt_home"])}


def data() -> dict:
    """Everything the NFL pages read, loaded once and reused until the database file changes."""
    stamp = DB_PATH.stat().st_mtime if DB_PATH.exists() else None
    if _cache.get("stamp") != stamp or "games" not in _cache:
        has_nfl = False
        if stamp is not None:
            with connect() as conn:
                has_nfl = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_game_info'").fetchone() is not None
        if not has_nfl:
            _cache.clear()
            _cache.update(empty_data(stamp))
            return _cache
        with connect() as conn:
            games = load_nfl_games(conn)
            extra = pd.read_sql("""SELECT i.game_id AS id, g.kickoff, i.source_game_id, i.total_line,
                                          i.home_moneyline, i.away_moneyline, i.stadium, i.overtime
                                   FROM nfl_game_info i JOIN games g ON g.id = i.game_id""", conn)
            games = add_qb_changes(games).merge(extra, on="id")
            live = load_live(conn).assign(source="Saved before kickoff")
            back = load_predictions(conn, BACKTEST_NAME, BACKTEST_VERSION).assign(source="Backtest")
            predictions = pd.concat([live, back[~back["id"].isin(live["id"])]], ignore_index=True)
            _cache.clear()
            _cache.update(stamp=stamp, games=games, predictions=predictions, backtest=back,
                          closing=load_closing(conn))
    return _cache


def finished_games() -> pd.DataFrame:
    g = data()["games"]
    return g[g["status"] == "final"]


def cached_model() -> dict:
    """Current ratings, fitted on finished games before today; refitted when the data or the date changes."""
    d, today = data(), date.today().isoformat()
    if d.get("model_day") != today:
        played = finished_games()
        start = (pd.Timestamp(today) - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
        train = played[(played["date"] >= start) & (played["date"] < today)]
        d["model"] = fit(train, today) if not train.empty else None
        if d["model"] is not None:
            d["model"]["games_used"] = len(train)
        d["model_day"] = today
    return d["model"]


def season_options(seasons, selected, url) -> list[dict]:
    return [{"label": str(s), "url": url(s), "selected": s == selected} for s in sorted(seasons, reverse=True)]


def render(request: Request, template: str, context: dict, status_code: int = 200):
    return templates.TemplateResponse(request, f"nfl/{template}", {"sport": "nfl", **context},
                                      status_code=status_code)


# ---------------------------------------------------------------- game cards

def game_card(g, current_team: str | None = None) -> dict:
    """Everything a game row needs: g has game columns plus (optionally) prediction and market columns."""
    finished = g.status == "final"
    home_margin_model = (g.exp_home - g.exp_away) if pd.notna(getattr(g, "exp_home", np.nan)) else None
    card = {
        "home": g.home, "away": g.away, "url": game_url(g.source_game_id),
        "date": day_label(g.date, with_year=True), "kickoff": kickoff_label(g.kickoff),
        "week": week_label(g.game_type, g.week), "playoff": g.game_type != "REG",
        "neutral": bool(g.neutral_site), "finished": finished,
        "model": None, "market": None, "source": getattr(g, "source", None),
        "model_line": line_text(home_margin_model, g.home, g.away),
        "model_total": f"{g.exp_home + g.exp_away:.1f}" if home_margin_model is not None else None,
        "market_line": line_text(g.spread_line, g.home, g.away),
        "market_total": f"{g.total_line:g}" if pd.notna(g.total_line) else None,
    }
    if pd.notna(getattr(g, "model_home", np.nan)):
        card["model"] = two_way(g.model_home)
    market_home = getattr(g, "mkt_home", np.nan)
    if pd.isna(market_home):
        market_home = no_vig(g.home_moneyline, g.away_moneyline)
    if market_home is not None and pd.notna(market_home):
        card["market"] = two_way(market_home)
    if finished:
        hs, as_ = int(g.home_score), int(g.away_score)
        card["score"] = f"{as_}–{hs}"   # away first, like the team names
        card["result"] = "home" if hs > as_ else "away" if as_ > hs else "tie"
        card["actual_line"] = line_text(hs - as_, g.home, g.away, decimals=0).replace("Even", "Tie")
        card["actual_total"] = hs + as_
        if card["model"] and card["result"] != "tie":
            card["model_actual"] = card["model"][card["result"]]
            card["market_actual"] = card["market"][card["result"]] if card["market"] else None
        if current_team:
            at_home = g.home == current_team
            card["outcome"] = outcome_letter(hs if at_home else as_, as_ if at_home else hs)
    if current_team:
        card["venue"] = "vs" if g.home == current_team else "at"
        card["opponent"] = g.away if g.home == current_team else g.home
    return card


def with_predictions(games: pd.DataFrame) -> pd.DataFrame:
    d = data()
    return games.merge(d["predictions"], on="id", how="left").merge(d["closing"], on="id", how="left")


def summary(df: pd.DataFrame) -> dict | None:
    """Typical probability for the actual winner, model vs market, on finished non-tied games that have both."""
    both = df[(df["status"] == "final") & (df["home_score"] != df["away_score"])].dropna(subset=["model_home", "mkt_home"])
    if both.empty:
        return None
    home_won = both["home_score"] > both["away_score"]
    model_p = np.where(home_won, both["model_home"], 1 - both["model_home"])
    market_p = np.where(home_won, both["mkt_home"], 1 - both["mkt_home"])
    return {"games": len(both),
            "model_typical": typical_probability(-np.log(model_p).mean()),
            "market_typical": typical_probability(-np.log(market_p).mean()),
            "model_favorite_won": int((model_p > 0.5).sum()),
            "model_closer": int((model_p > market_p).sum())}


# ---------------------------------------------------------------- pages

@router.get("/", response_class=HTMLResponse)
def upcoming(request: Request):
    d = data()
    today = date.today().isoformat()
    live = d["predictions"][d["predictions"]["source"] == "Saved before kickoff"]
    games = (d["games"][(d["games"]["status"] == "scheduled") & (d["games"]["date"] >= today)]
             .merge(live, on="id").sort_values(["date", "kickoff", "home"]))
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]}
            for day, group in games.groupby("date", sort=True)]
    week = week_label(games.iloc[0]["game_type"], int(games.iloc[0]["week"])) if len(games) else None
    return render(request, "upcoming.html", {"active": "upcoming", "days": days, "week": week})


def week_key(season: int, week: int) -> str:
    return f"{BASE}/nfl/results/{season}/{week}/"


@router.get("/results/", response_class=HTMLResponse)
def results_latest(request: Request):
    return results_page(request, None, None)


@router.get("/results/{season}/{week}/", response_class=HTMLResponse)
def results_week(request: Request, season: int, week: int):
    return results_page(request, season, week)


def results_page(request: Request, season: int | None, week: int | None):
    played = with_predictions(finished_games()).dropna(subset=["model_home"])
    if played.empty:
        return render(request, "results.html", {"active": "results", "days": []})
    weeks = sorted(set(zip(played["season"], played["week"])))
    if season is None:
        season, week = weeks[-1]
    elif (season, week) not in weeks:
        raise HTTPException(status_code=404)
    position = weeks.index((season, week))
    wk = played[(played["season"] == season) & (played["week"] == week)].sort_values(["date", "kickoff", "home"])
    days = [{"label": day_label(day), "games": [game_card(g) for g in group.itertuples()]}
            for day, group in wk.groupby("date", sort=True)]
    return render(request, "results.html", {
        "active": "results", "days": days, "summary": summary(wk),
        "week_label": f"{season} season, {week_label(wk.iloc[0]['game_type'], week)}",
        "previous_url": week_key(*weeks[position - 1]) if position > 0 else None,
        "next_url": week_key(*weeks[position + 1]) if position < len(weeks) - 1 else None,
    })


def standings(season: int) -> pd.DataFrame:
    """Regular-season records for one season: wins, losses, ties, points for and against, last 5."""
    g = finished_games()
    g = g[(g["season"] == season) & (g["game_type"] == "REG")]
    rows = pd.concat([
        pd.DataFrame({"team": g["home"], "date": g["date"], "pf": g["home_score"], "pa": g["away_score"]}),
        pd.DataFrame({"team": g["away"], "date": g["date"], "pf": g["away_score"], "pa": g["home_score"]}),
    ], ignore_index=True)
    rows[["pf", "pa"]] = rows[["pf", "pa"]].astype(int)
    rows["outcome"] = [outcome_letter(pf, pa) for pf, pa in zip(rows["pf"], rows["pa"])]
    table = rows.groupby("team").agg(
        won=("outcome", lambda s: (s == "W").sum()), lost=("outcome", lambda s: (s == "L").sum()),
        tied=("outcome", lambda s: (s == "T").sum()), pf=("pf", "sum"), pa=("pa", "sum"))
    table["played"] = table["won"] + table["lost"] + table["tied"]
    table["pct"] = (table["won"] + 0.5 * table["tied"]) / table["played"]
    table["diff"] = table["pf"] - table["pa"]
    table["form"] = rows.sort_values("date").groupby("team")["outcome"].apply(lambda s: list(s.tail(5)))
    table["division"] = [TEAMS.get(t, ("", ""))[1] for t in table.index]
    return table.reset_index().sort_values(["pct", "diff", "pf"], ascending=False).reset_index(drop=True)


def record_text(r) -> str:
    return f"{r.won}–{r.lost}" + (f"–{r.tied}" if r.tied else "")


@router.get("/standings/", response_class=HTMLResponse)
def standings_latest(request: Request):
    return standings_page(request, None)


@router.get("/standings/{season}/", response_class=HTMLResponse)
def standings_season(request: Request, season: int):
    return standings_page(request, season)


def standings_page(request: Request, season: int | None):
    played = finished_games()
    seasons = sorted(played.loc[played["game_type"] == "REG", "season"].unique())
    if not seasons:
        return render(request, "standings.html", {"active": "standings", "groups": []})
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    table = standings(season)

    def row(r):
        return {"team": r.team, "name": team_name(r.team), "url": team_url(r.team, season), "record": record_text(r),
                "won": r.won, "lost": r.lost, "tied": r.tied, "pct": f"{r.pct:.3f}".lstrip("0"),
                "pf": r.pf, "pa": r.pa, "diff": signed_number(r.diff), "form": r.form}

    if season >= DIVISIONS_FROM:
        groups = []
        for conference in ("AFC", "NFC"):
            for division in ("East", "North", "South", "West"):
                name = f"{conference} {division}"
                members = table[table["division"] == name]
                groups.append({"label": name, "rows": [row(r) for r in members.itertuples()]})
    else:
        groups = [{"label": "All teams", "rows": [row(r) for r in table.itertuples()]}]
    return render(request, "standings.html", {
        "active": "standings", "groups": groups, "season": season, "is_current": season == seasons[-1],
        "has_ties": bool(table["tied"].sum()), "by_division": season >= DIVISIONS_FROM,
        "seasons": season_options(seasons, season, lambda s: f"{BASE}/nfl/standings/{s}/"),
    })


def find_team(slug: str) -> str:
    teams = {slugify(team_name(t)): t for t in pd.concat([data()["games"]["home"], data()["games"]["away"]]).unique()}
    if slug not in teams:
        raise HTTPException(status_code=404)
    return teams[slug]


def rating_row(model: dict, team: str) -> dict | None:
    if model is None or team not in model["ratings"].index:
        return None
    r = model["ratings"].loc[team]
    return {"offense": signed_number(r["offense"], 1), "defense": signed_number(r["defense"], 1),
            "overall": signed_number(r["offense"] + r["defense"], 1),
            "offense_class": "good" if r["offense"] > 0 else "bad" if r["offense"] < 0 else "",
            "defense_class": "good" if r["defense"] > 0 else "bad" if r["defense"] < 0 else "",
            "overall_class": "good" if r["offense"] + r["defense"] > 0 else "bad"}


@router.get("/team/{slug}/", response_class=HTMLResponse)
def team_latest(request: Request, slug: str):
    return team_page(request, slug, None)


@router.get("/team/{slug}/{season}/", response_class=HTMLResponse)
def team_season(request: Request, slug: str, season: int):
    return team_page(request, slug, season)


def team_page(request: Request, slug: str, season: int | None):
    team = find_team(slug)
    games = data()["games"]
    own = games[(games["home"] == team) | (games["away"] == team)]
    seasons = sorted(own["season"].unique())
    if season is None:
        season = seasons[-1]
    elif season not in seasons:
        raise HTTPException(status_code=404)
    latest = games["season"].max()
    shown = with_predictions(own[own["season"] == season]).sort_values("date")
    played = shown[shown["status"] == "final"]
    coming = shown[shown["status"] == "scheduled"]

    record = None
    table = standings(season)
    if team in set(table["team"]):
        r = table.set_index("team").loc[team]
        division = r["division"]
        div_rank = None
        if season >= DIVISIONS_FROM:
            div_teams = table[table["division"] == division]["team"].tolist()
            div_rank = ordinal(div_teams.index(team) + 1)
        record = {"text": record_text(r), "pf": int(r["pf"]), "pa": int(r["pa"]), "diff": signed_number(r["diff"]),
                  "division": division if season >= DIVISIONS_FROM else None, "div_rank": div_rank}

    qbs = (pd.concat([played.loc[played["home"] == team, "home_qb"], played.loc[played["away"] == team, "away_qb"]])
           .dropna().value_counts())
    return render(request, "team.html", {
        "active": None, "team": team, "name": team_name(team), "division": TEAMS.get(team, ("", ""))[1],
        "season": season, "is_current": season == latest, "record": record,
        "ratings": rating_row(cached_model(), team) if season == latest else None,
        "quarterbacks": [{"name": q, "starts": int(n)} for q, n in qbs.items()],
        "summary": summary(played),
        "played": [game_card(g, team) for g in played.sort_values("date", ascending=False).itertuples()],
        "coming": [game_card(g, team) for g in coming.dropna(subset=["model_home"]).itertuples()],
        "schedule": [{"date": day_label(g.date, with_year=True), "week": week_label(g.game_type, g.week),
                      "venue": "vs" if g.home == team else "at", "opponent": g.away if g.home == team else g.home,
                      "kickoff": kickoff_label(g.kickoff), "url": game_url(g.source_game_id)}
                     for g in coming[coming["model_home"].isna()].itertuples()],
        "standings_url": f"{BASE}/nfl/standings/{season}/",
        "seasons": season_options(seasons, season, lambda s: team_url(team, s)),
    })


def form(games: pd.DataFrame, team: str, before: str, n: int = 5) -> list[dict]:
    """A team's last n games before a date, oldest first."""
    own = games[((games["home"] == team) | (games["away"] == team)) & (games["date"] < before)
                & (games["status"] == "final")]
    rows = []
    for g in own.sort_values("date").tail(n).itertuples():
        at_home = g.home == team
        ts, os_ = (g.home_score, g.away_score) if at_home else (g.away_score, g.home_score)
        rows.append({"outcome": outcome_letter(ts, os_), "score": f"{int(ts)}–{int(os_)}",
                     "venue": "vs" if at_home else "at", "opponent": g.away if at_home else g.home,
                     "url": game_url(g.source_game_id)})
    return rows


def head_to_head(games: pd.DataFrame, a: str, b: str, before: str) -> dict:
    """Every earlier meeting of two teams (since 1999), most recent first, with the series record."""
    met = games[(((games["home"] == a) & (games["away"] == b)) | ((games["home"] == b) & (games["away"] == a)))
                & (games["date"] < before) & (games["status"] == "final")].sort_values("date", ascending=False)
    wins = {a: 0, b: 0}
    ties = 0
    meetings = []
    for g in met.itertuples():
        winner = g.home if g.home_score > g.away_score else g.away if g.away_score > g.home_score else None
        if winner:
            wins[winner] += 1
        else:
            ties += 1
        meetings.append({"date": day_label(g.date, with_year=True), "season": g.season,
                         "week": week_label(g.game_type, g.week), "home": g.home, "away": g.away,
                         "score": f"{int(g.away_score)}–{int(g.home_score)}", "winner": winner,
                         "url": game_url(g.source_game_id)})
    return {"meetings": meetings, "wins_a": wins[a], "wins_b": wins[b], "ties": ties,
            "a": a, "b": b, "first": int(met["season"].min()) if len(met) else None}


def player_projections(game_id: int, finished: bool) -> list[dict] | None:
    """Each team's projected players for a game (quarterback first, then by yards), with the actual
    numbers next to them once it's been played. None if the game has no projections."""
    with connect() as conn:
        if conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_player_projections'").fetchone() is None:
            return None
        proj = pd.read_sql("SELECT * FROM nfl_player_projections WHERE game_id = ?", conn, params=(game_id,))
        actual = pd.read_sql("SELECT * FROM nfl_player_stats WHERE game_id = ?", conn, params=(game_id,)) \
            if finished else pd.DataFrame()
    if proj.empty:
        return None
    if not actual.empty:
        actual_columns = ["passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
                          "receptions", "receiving_yards", "receiving_tds"]
        proj = proj.merge(actual[["player_id", *actual_columns]].rename(columns={c: f"act_{c}" for c in actual_columns}),
                          on="player_id", how="left")
    fmt = lambda v, d=0: "" if v is None or pd.isna(v) else f"{v:.{d}f}"
    from nfl.players import load_ranges, stat_range
    ranges = load_ranges() if not finished else None

    def span(stat, value):
        """'15–95': the 80% range, for upcoming games."""
        if not ranges or value is None or pd.isna(value) or value < 0.5:
            return None
        low, high = stat_range(ranges, stat, value)
        return f"{low:.0f}\u2013{high:.0f}"

    teams = []
    for team, rows in proj.groupby("team"):
        rows = rows.assign(order=(rows["position"] != "QB").astype(int),
                           yards=rows["pass_yds"] + rows["rush_yds"] + rows["rec_yds"])
        rows = rows.sort_values(["order", "yards"], ascending=[True, False])
        players = []
        for r in rows.itertuples():
            if r.position != "QB" and r.rush_yds + r.rec_yds < 5:
                continue   # too small a role to be worth listing
            row = {"name": r.name, "position": r.position, "qb": r.position == "QB",
                   "pass_yds": fmt(r.pass_yds), "pass_tds": fmt(r.pass_tds, 1), "ints": fmt(r.ints, 1),
                   "rush_yds": fmt(r.rush_yds), "receptions": fmt(r.receptions, 1), "rec_yds": fmt(r.rec_yds),
                   "anytime_td": f"{r.anytime_td * 100:.0f}%", "actual": None,
                   "ranges": {stat: span(stat, getattr(r, stat)) for stat in ("pass_yds", "rush_yds", "receptions", "rec_yds")}}
            if "act_receiving_yards" in rows:
                played = not pd.isna(r.act_rushing_yards)
                tds = (r.act_rushing_tds or 0) + (r.act_receiving_tds or 0) if played else None
                row["actual"] = {"pass_yds": fmt(r.act_passing_yards), "pass_tds": fmt(r.act_passing_tds),
                                 "ints": fmt(r.act_interceptions), "rush_yds": fmt(r.act_rushing_yards),
                                 "receptions": fmt(r.act_receptions), "rec_yds": fmt(r.act_receiving_yards),
                                 "scored": None if tds is None else bool(tds > 0), "played": played}
            players.append(row)
        teams.append({"team": team, "players": players})
    return teams


@router.get("/game/{key}/", response_class=HTMLResponse)
def game_page(request: Request, key: str):
    d = data()
    games = d["games"]
    found = games[games["source_game_id"].str.lower().str.replace("_", "-") == key]
    if found.empty:
        raise HTTPException(status_code=404)
    g = next(with_predictions(found).itertuples())
    card = game_card(g)
    finished = g.status == "final"
    qb = None
    if isinstance(g.home_qb_expected, str) or isinstance(g.away_qb_expected, str):
        model = cached_model() if not finished else None   # current ratings only describe upcoming games
        value = lambda side: (signed_number(qb_value(model, getattr(g, f"{side}_qb_expected"),
                                                     getattr(g, f"{side}_qb_starts")), 1) if model else None)
        qb = {side: {"name": getattr(g, f"{side}_qb_expected"), "listed": isinstance(getattr(g, f"{side}_qb"), str),
                     "change": bool(getattr(g, f"{side}_qb_change")), "value": value(side),
                     "starts": None if pd.isna(getattr(g, f"{side}_qb_starts")) else int(getattr(g, f"{side}_qb_starts"))}
              for side in ("away", "home")}
    expected = None
    if pd.notna(getattr(g, "exp_home", np.nan)):
        expected = {"home": f"{g.exp_home:.1f}", "away": f"{g.exp_away:.1f}"}
    return render(request, "game.html", {
        "active": None, "g": card, "finished": finished, "expected": expected, "qb": qb,
        "date": day_label(g.date), "season": int(g.season), "stadium": g.stadium,
        "home_score": int(g.home_score) if finished else None, "away_score": int(g.away_score) if finished else None,
        "form_home": form(games, g.home, g.date), "form_away": form(games, g.away, g.date),
        "h2h": head_to_head(games, g.away, g.home, g.date),
        "margin_sd": f"{MARGIN_SD:g}",
        "players": player_projections(int(g.id), finished),
    })


def live_scores(df: pd.DataFrame) -> dict:
    """The model's record (and the closing market's, where available) on finished live predictions."""
    decided = df[df["home_score"] != df["away_score"]]
    home_won = decided["home_score"] > decided["away_score"]
    model_p = np.where(home_won, decided["model_home"], 1 - decided["model_home"])
    priced = decided.dropna(subset=["mkt_home"])
    priced_won = priced["home_score"] > priced["away_score"]
    market_p = np.where(priced_won, priced["mkt_home"], 1 - priced["mkt_home"])
    model_on_priced = np.where(priced_won, priced["model_home"], 1 - priced["model_home"])
    return {"games": len(decided), "ties": len(df) - len(decided),
            "model_typical": typical_probability(-np.log(model_p).mean()) if len(decided) else None,
            "favorite_won": int((model_p > 0.5).sum()),
            "compared": len(priced),
            "model_typical_compared": typical_probability(-np.log(model_on_priced).mean()) if len(priced) else None,
            "market_typical": typical_probability(-np.log(market_p).mean()) if len(priced) else None}


STAT_LABELS = {"pass_yds": "Passing yards", "pass_tds": "Passing touchdowns", "rush_yds": "Rushing yards",
               "receptions": "Receptions", "rec_yds": "Receiving yards"}


def player_track() -> dict | None:
    """Live player projections on finished games against what happened and the simple benchmarks."""
    from nfl.players import STATS, accuracy, compare
    d = data()
    if "player_track" in d:
        return d["player_track"]
    result = None
    with connect() as conn:
        has = conn.execute("SELECT name FROM sqlite_master WHERE name = 'nfl_player_projections'").fetchone()
        proj = pd.read_sql("""SELECT p.* FROM nfl_player_projections p JOIN games g ON g.id = p.game_id
                              WHERE g.status = 'final'""", conn) if has else pd.DataFrame()
        if not proj.empty:
            games = pd.read_sql("""SELECT g.id, g.season, g.date, i.week, i.game_type FROM games g
                                   JOIN nfl_game_info i ON i.game_id = g.id""", conn)
            ids = sorted(set(proj["player_id"]))
            players = pd.concat([pd.read_sql(f"SELECT * FROM nfl_player_stats WHERE player_id IN "
                                             f"({','.join('?' * len(ids[i:i + 500]))})", conn, params=ids[i:i + 500])
                                 for i in range(0, len(ids), 500)], ignore_index=True)
    if not proj.empty:
        columns = ["game_id", "player_id", *STATS, "anytime_td"]
        df = compare(proj[columns], players, games)
        missing = len(proj) - len(df)
        acc = accuracy(df).set_index("stat")
        overall = [{"label": STAT_LABELS[s], "players": int(acc.at[s, "players"]),
                    "model": f"{acc.at[s, 'model']:.{2 if s in ('pass_tds', 'receptions') else 1}f}",
                    "last3": f"{acc.at[s, 'last 3 games']:.{2 if s in ('pass_tds', 'receptions') else 1}f}",
                    "season": f"{acc.at[s, 'season average']:.{2 if s in ('pass_tds', 'receptions') else 1}f}",
                    "best": acc.at[s, "model"] <= min(acc.at[s, "last 3 games"], acc.at[s, "season average"])}
                   for s in STATS if acc.at[s, "players"] > 0]

        df = df.merge(games[["id", "week", "game_type"]].rename(columns={"id": "game_id"}), on="game_id")
        df["proj_total"] = df["proj_pass_yds"] + df["proj_rush_yds"] + df["proj_rec_yds"]
        df["actual_total"] = df[["passing_yards", "rushing_yards", "receiving_yards"]].fillna(0).sum(axis=1)
        weeks = []
        for (season, week), part in sorted(df.groupby(["season", "week"]), key=lambda kv: kv[0], reverse=True):
            miss = lambda col, actual: (part[col] - part[actual].fillna(0)).abs().mean()
            weeks.append({"label": f"{season} {week_label(part.iloc[0]['game_type'], week)}", "players": len(part),
                          "rec_model": f"{miss('proj_rec_yds', 'receiving_yards'):.1f}",
                          "rec_season": f"{miss('season_rec_yds', 'receiving_yards'):.1f}",
                          "rush_model": f"{miss('proj_rush_yds', 'rushing_yards'):.1f}",
                          "rush_season": f"{miss('season_rush_yds', 'rushing_yards'):.1f}"})

        scored = (df["rushing_tds"].fillna(0) + df["receiving_tds"].fillna(0)) > 0
        bins = pd.cut(df["proj_anytime_td"], [0, 0.15, 0.3, 0.45, 1])
        calibration = [{"range": f"{int(b.left * 100)}\u2013{int(b.right * 100)}%", "players": len(part),
                        "predicted": f"{part['proj_anytime_td'].mean() * 100:.0f}%",
                        "actual": f"{scored[part.index].mean() * 100:.0f}%"}
                       for b, part in df.groupby(bins, observed=True)]

        latest = df[(df["season"] == df["season"].max())]
        latest = latest[latest["week"] == latest["week"].max()].copy()
        latest["miss"] = latest["actual_total"] - latest["proj_total"]
        surprises = [{"name": r.name, "team": r.team, "projected": f"{r.proj_total:.0f}", "actual": f"{r.actual_total:.0f}",
                      "url": game_url(data()["games"].set_index("id").at[r.game_id, "source_game_id"])}
                     for r in latest.reindex(latest["miss"].abs().sort_values(ascending=False).index).head(6).itertuples()]
        result = {"overall": overall, "weeks": weeks, "calibration": calibration, "surprises": surprises,
                  "surprise_week": weeks[0]["label"] if weeks else None, "games": int(df["game_id"].nunique()),
                  "missing": missing}
    d["player_track"] = result
    return result


@router.get("/track-record/", response_class=HTMLResponse)
def track_record(request: Request):
    d = data()
    live = d["predictions"][d["predictions"]["source"] == "Saved before kickoff"]
    df = (finished_games().merge(live, on="id").merge(d["closing"], on="id", how="left")
          .sort_values(["date", "kickoff", "home"]))
    context = {"active": "track", "record_url": RECORD_URL, "history_url": RECORD_HISTORY_URL, "weeks": [],
               "players": player_track(), "player_record_url": PLAYER_RECORD_URL}
    if df.empty:
        return render(request, "track_record.html", context)
    weeks = []
    for (season, week), group in sorted(df.groupby(["season", "week"]), key=lambda kv: kv[0], reverse=True):
        weeks.append({"label": f"{season} {week_label(group.iloc[0]['game_type'], week)}",
                      "scores": live_scores(group), "games": [game_card(g) for g in group.itertuples()]})
    held = backtest_frame()
    long_run = typical_probability(held["model_logloss"].mean()) if held is not None else None
    return render(request, "track_record.html", {**context, "overall": live_scores(df), "weeks": weeks,
                                                 "first_date": day_label(df["date"].min(), with_year=True),
                                                 "long_run": long_run})


def backtest_frame() -> pd.DataFrame | None:
    """Held-out backtest games (2018 onward, ties left out) with model and market log loss."""
    d = data()
    df = (finished_games().merge(d["backtest"], on="id").merge(d["closing"], on="id")
          .query("home_score != away_score"))
    if df.empty:
        return None
    won = df["home_score"] > df["away_score"]
    df["model_logloss"] = -np.log(np.where(won, df["model_home"], 1 - df["model_home"]))
    df["mkt_logloss"] = -np.log(np.where(won, df["mkt_home"], 1 - df["mkt_home"]))
    return df


@router.get("/backtest/", response_class=HTMLResponse)
def backtest_page(request: Request):
    df = backtest_frame()
    if df is None:
        return render(request, "backtest.html", {"active": "backtest", "seasons": []})
    by_season = (df.groupby("season").agg(games=("id", "size"), model=("model_logloss", "mean"),
                                           market=("mkt_logloss", "mean")).reset_index())
    by_season["gap"] = by_season["model"] - by_season["market"]
    widest = by_season["gap"].abs().max() or 1
    seasons = [{"label": str(r.season), "games": r.games, "model": f"{r.model:.3f}", "market": f"{r.market:.3f}",
                "gap": f"{abs(r.gap):.3f}", "market_better": r.gap > 0,
                "width": round(abs(r.gap) / widest * 100, 1), "held_out": r.season >= HELD_OUT_FROM}
               for r in by_season.itertuples()]

    held = df[df["season"] >= HELD_OUT_FROM].copy()
    margin = held["home_score"] - held["away_score"]
    held_all = finished_games().merge(data()["backtest"], on="id")
    held_all = held_all[held_all["season"] >= HELD_OUT_FROM]
    actual_margin = held_all["home_score"] - held_all["away_score"]
    model_miss = (actual_margin - (held_all["exp_home"] - held_all["exp_away"])).abs().mean()
    spread_miss = (actual_margin - held_all["spread_line"]).abs().mean()

    # Calibration: when the model gave the home team a given chance, how often did it win?
    bins = pd.cut(held["model_home"], [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1])
    calibration = [{"range": f"{int(b.left * 100)}–{int(b.right * 100)}%", "games": len(part),
                    "predicted": f"{part['model_home'].mean() * 100:.0f}%",
                    "actual": f"{(part['home_score'] > part['away_score']).mean() * 100:.0f}%"}
                   for b, part in held.groupby(bins, observed=True)]
    return render(request, "backtest.html", {
        "active": "backtest", "seasons": seasons,
        "headline": {"games": len(held), "from": HELD_OUT_FROM,
                     "model": f"{held['model_logloss'].mean():.4f}", "market": f"{held['mkt_logloss'].mean():.4f}",
                     "model_typical": typical_probability(held["model_logloss"].mean()),
                     "market_typical": typical_probability(held["mkt_logloss"].mean()),
                     "picked": round(((held["model_home"] > 0.5) == (margin > 0)).mean() * 100),
                     "market_picked": round(((held["mkt_home"] > 0.5) == (margin > 0)).mean() * 100)},
        "margins": {"model": f"{model_miss:.1f}", "spread": f"{spread_miss:.1f}"},
        "calibration": calibration,
        "development": [{"label": label, "value": f"{value:.4f}"} for label, value in DEVELOPMENT],
        "player_accuracy": PLAYER_ACCURACY,
        "start": START_SEASON, "tuned_through": HELD_OUT_FROM - 1, "half_life": HALF_LIFE_DAYS,
        "training_years": TRAINING_YEARS,
    })


@router.get("/ratings/", response_class=HTMLResponse)
def ratings(request: Request):
    model = cached_model()
    games = data()["games"]
    if model is None:
        return render(request, "ratings.html", {"active": "ratings", "rows": []})
    current = games[games["season"] == games["season"].max()]
    teams = sorted(set(current["home"]) | set(current["away"]))
    r = model["ratings"].loc[[t for t in teams if t in model["ratings"].index]].copy()
    r["overall"] = r["offense"] + r["defense"]
    rows = [{"team": t, "name": team_name(t), "division": TEAMS.get(t, ("", ""))[1], **rating_row(model, t)}
            for t in r.sort_values("overall", ascending=False).index]
    return render(request, "ratings.html", {
        "active": "ratings", "rows": rows, "as_of": day_label(date.today().isoformat()),
        "games_used": model["games_used"], "home_adv": f"{model['home_adv']:.1f}",
        "inexperience": f"{abs(model['inexperience']):.1f}", "inexperience_starts": INEXPERIENCE_STARTS,
        "average_points": f"{model['base']:.1f}", "quarterbacks": quarterback_rows(model, teams),
    })


def quarterback_rows(model: dict, teams: list[str]) -> list[dict]:
    """Each current team's expected starter for its next game (or its latest starter), best first."""
    games = data()["games"]
    rows = []
    for team in teams:
        own = games[(games["home"] == team) | (games["away"] == team)].sort_values("date")
        upcoming = own[own["status"] == "scheduled"]
        g = (upcoming if len(upcoming) else own).iloc[0 if len(upcoming) else -1]
        side = "home" if g["home"] == team else "away"
        qb, starts = g[f"{side}_qb_expected"], g[f"{side}_qb_starts"]
        if not isinstance(qb, str):
            continue
        value = qb_value(model, qb, starts)
        rows.append({"team": team, "name": qb, "value": signed_number(value, 1), "raw": value,
                     "class": "good" if value > 0 else "bad" if value < 0 else "",
                     "starts": None if pd.isna(starts) else int(starts)})
    return sorted(rows, key=lambda r: -r["raw"])
