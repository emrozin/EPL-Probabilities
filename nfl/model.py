"""Points model for the NFL: offense and defense ratings, fitted like the Premier League model.

Each team has an offense rating (points it scores above average) and a defense rating
(points it stops). A team's expected points in a game:

    home points = base + home advantage / 2 + home offense - away defense
    away points = base - home advantage / 2 + away offense - home defense

so the expected home margin is  home advantage + (home offense + home defense) - (away offense + away defense).

Ratings are fitted by weighted least squares on recent games (each game's weight halves every
`half_life_days`, so last season counts less than this one), with shrinkage pulling ratings
toward average so a few games can't produce extreme ratings. Home advantage is fitted from
the same recent games, so it follows the league's shrinking home edge on its own.

The actual margin lands around the expected one with a typical spread of `margin_sd` points,
so the home win probability is P(margin > 0) from a normal distribution. Unlike Elo, this also
gives an expected score for each team, so its own spread and total.

Quarterbacks: a team whose starting QB isn't its usual starter (the most frequent starter in
its previous 8 games) scores fewer points than its ratings suggest, whether it's an injury
backup or a new starter. The model fits that effect, in points, alongside the ratings
(`qb_change` below), so it's learned only from earlier games like everything else.
"""

from collections import Counter


import numpy as np
import pandas as pd
from scipy.stats import norm

# Tuned on 2010-2017 (nfl/tune_model.py)
HALF_LIFE_DAYS = 150  # a game's weight halves every 150 days, so last season counts about half as much
SHRINKAGE = 3.0       # pull toward average ratings (L2 regularization)
MARGIN_SD = 13.0      # typical gap between the actual and expected margin, in points
TRAINING_YEARS = 4


USUAL_STARTER_GAMES = 8


def add_qb_changes(games: pd.DataFrame) -> pd.DataFrame:
    """Add home_qb_change / away_qb_change: 1 if that side's starting QB isn't its usual starter.

    The usual starter is the most frequent starter in the team's previous 8 games, so this only
    looks backward. Games whose starter isn't known (not played yet, or missing) get 0.
    """
    games = games.copy()
    history: dict[str, list] = {}
    flags = {"home": {}, "away": {}}
    for g in games.sort_values(["date", "id"]).itertuples():
        for side in ("home", "away"):
            team, qb = getattr(g, side), getattr(g, f"{side}_qb")
            qb = qb if isinstance(qb, str) and qb else None   # missing can arrive as None or NaN
            recent = [q for q in history.get(team, [])[-USUAL_STARTER_GAMES:] if q]
            usual = Counter(recent).most_common(1)[0][0] if recent else None
            flags[side][g.id] = int(qb is not None and usual is not None and qb != usual)
        for side in ("home", "away"):
            qb = getattr(g, f"{side}_qb")
            if g.status == "final":
                history.setdefault(getattr(g, side), []).append(qb if isinstance(qb, str) and qb else None)
    for side in ("home", "away"):
        games[f"{side}_qb_change"] = games["id"].map(flags[side])
    return games


def fit(train: pd.DataFrame, as_of, half_life_days=HALF_LIFE_DAYS, shrinkage=SHRINKAGE) -> dict:
    """Fit offense/defense ratings, base points and home advantage from finished games before `as_of`."""
    teams = sorted(set(train["home"]) | set(train["away"]))
    index = {t: i for i, t in enumerate(teams)}
    n, games = len(teams), len(train)

    days_ago = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    w = np.tile(0.5 ** (days_ago / half_life_days), 2)

    # Two rows per game: the home team's points and the away team's points.
    # Columns: offense (n), defense (n), base, home advantage, QB change.
    x = np.zeros((2 * games, 2 * n + 3))
    h = train["home"].map(index).to_numpy()
    a = train["away"].map(index).to_numpy()
    rows = np.arange(games)
    home_side = np.where(train["neutral_site"].to_numpy() == 1, 0.0, 0.5)
    x[rows, h] = 1                 # home offense
    x[rows, n + a] = -1            # away defense
    x[rows, 2 * n] = 1
    x[rows, 2 * n + 1] = home_side
    x[games + rows, a] = 1         # away offense
    x[games + rows, n + h] = -1    # home defense
    x[games + rows, 2 * n] = 1
    x[games + rows, 2 * n + 1] = -home_side
    x[rows, 2 * n + 2] = train["home_qb_change"].to_numpy(float)
    x[games + rows, 2 * n + 2] = train["away_qb_change"].to_numpy(float)
    y = np.concatenate([train["home_score"].to_numpy(float), train["away_score"].to_numpy(float)])

    penalty = np.zeros(2 * n + 3)
    penalty[:2 * n] = shrinkage    # shrink team ratings toward 0 (average), not base or home advantage
    penalty[2 * n + 2] = 1e-6      # keeps the QB effect solvable (at 0) when no game in the window has a QB change
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x + np.diag(penalty), xtw @ y)
    offense, defense = beta[:n] - beta[:n].mean(), beta[n:2 * n] - beta[n:2 * n].mean()
    return {"ratings": pd.DataFrame({"offense": offense, "defense": defense}, index=teams),
            "base": beta[2 * n] + beta[:n].mean() - beta[n:2 * n].mean(),
            "home_adv": beta[2 * n + 1],
            "qb_change": beta[2 * n + 2]}   # points a team scores when its QB isn't its usual starter (negative)


def predict(model: dict, home: str, away: str, neutral: bool = False, margin_sd=MARGIN_SD,
            home_qb_change: int = 0, away_qb_change: int = 0) -> dict:
    """Expected points for each side, the expected home margin and total, and the home win probability."""
    r = model["ratings"]
    edge = 0.0 if neutral else model["home_adv"] / 2
    home_pts = (model["base"] + edge + r.at[home, "offense"] - r.at[away, "defense"]
                + model["qb_change"] * home_qb_change)
    away_pts = (model["base"] - edge + r.at[away, "offense"] - r.at[home, "defense"]
                + model["qb_change"] * away_qb_change)
    margin = home_pts - away_pts
    return {"home_points": home_pts, "away_points": away_pts, "margin": margin, "total": home_pts + away_pts,
            "home_win": float(norm.cdf(margin / margin_sd))}


def backtest(games: pd.DataFrame, start_season: int, half_life_days=HALF_LIFE_DAYS, shrinkage=SHRINKAGE,
             margin_sd=MARGIN_SD, training_years=TRAINING_YEARS) -> pd.DataFrame:
    """Walk-forward: for each week of each season, fit on earlier games only, then predict that week.
    `games` needs the QB change columns (add_qb_changes)."""
    played = games[games["status"] == "final"]
    test = played[played["season"] >= start_season]
    rows = []
    for (season, week), week_games in test.groupby(["season", "week"], sort=True):
        cutoff = week_games["date"].min()
        start = (pd.Timestamp(cutoff) - pd.DateOffset(years=training_years)).strftime("%Y-%m-%d")
        train = played[(played["date"] >= start) & (played["date"] < cutoff)]
        if train.empty:
            continue
        model = fit(train, cutoff, half_life_days, shrinkage)
        known = set(model["ratings"].index)
        for g in week_games.itertuples():
            if g.home not in known or g.away not in known:
                continue
            p = predict(model, g.home, g.away, bool(g.neutral_site), margin_sd,
                        g.home_qb_change, g.away_qb_change)
            rows.append({"id": g.id, "model_home": p["home_win"], "model_margin": p["margin"],
                         "model_total": p["total"], "home_points": p["home_points"],
                         "away_points": p["away_points"]})
    return pd.DataFrame(rows)
