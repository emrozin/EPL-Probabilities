"""College football points model: offense and defense ratings for every team, like the NFL's
(nfl/model.py) but without quarterback ratings, which the free data can't support.

Each game gives two equations: home points = base + home edge + home offense - away defense,
and the same for the away team. Ratings are fitted by weighted least squares on recent seasons:
recent games count more (HALF_LIFE_DAYS), and games from earlier seasons count less again
(SEASON_WEIGHT per season back), because college rosters turn over far more than NFL ones.
Ratings are pulled toward average (SHRINKAGE), which matters most for teams with few games
against the rest, like second-division (FCS) schools. The expected margin becomes a win
probability through a normal distribution (MARGIN_SD).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# Tuned on 2010-2017 (python -m cfb.tune_model): log loss 0.536 with first guesses (half-life 200,
# season weight 0.6, shrinkage 4, 3 years) down to 0.479. Much weaker shrinkage than the NFL's
# mattered most: college teams really are far apart, and with old games down-weighted, strong
# shrinkage pulled every rating toward average.
HALF_LIFE_DAYS = 400
SEASON_WEIGHT = 0.4       # each earlier season's games count this much less again
SHRINKAGE = 0.05
MARGIN_SD = 16.0
TRAINING_YEARS = 4


def load_games(conn) -> pd.DataFrame:
    """Every college game (played and scheduled), in date order."""
    return pd.read_sql("""
        SELECT g.id, g.season, g.date, g.kickoff, g.status, h.name AS home, a.name AS away,
               g.home_score, g.away_score, i.neutral_site, i.week, i.season_type, i.source_game_id,
               i.home_division, i.away_division, i.home_conference, i.away_conference,
               i.home_elo, i.away_elo, i.spread_line, i.total_line, i.notes, i.venue, i.conference_game
        FROM games g
        JOIN leagues l ON l.id = g.league_id AND l.code = 'CFB'
        JOIN teams h ON h.id = g.home_team_id
        JOIN teams a ON a.id = g.away_team_id
        JOIN cfb_game_info i ON i.game_id = g.id
        ORDER BY g.date, g.id
    """, conn)


def load_market(conn) -> pd.DataFrame:
    """The consensus closing moneyline (margin removed) for finished games: market_home."""
    return pd.read_sql("""SELECT game_id AS id, 1 / price AS market_home FROM odds
                          WHERE bookmaker = 'consensus' AND market = 'moneyline' AND outcome = 'home'""", conn)


def fit(train: pd.DataFrame, as_of: str, season: int | None = None, half_life_days=HALF_LIFE_DAYS,
        season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE) -> dict:
    """Offense/defense ratings, base points and home advantage from finished games before as_of.
    season: the season being predicted (earlier seasons' games are down-weighted)."""
    teams = sorted(set(train["home"]) | set(train["away"]))
    index = {t: i for i, t in enumerate(teams)}
    n, games = len(teams), len(train)
    days_ago = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    seasons_back = (season - train["season"].to_numpy()) if season is not None else 0
    w = np.tile(0.5 ** (days_ago / half_life_days) * season_weight ** np.maximum(seasons_back, 0), 2)

    x = np.zeros((2 * games, 2 * n + 2))
    h = train["home"].map(index).to_numpy()
    a = train["away"].map(index).to_numpy()
    rows = np.arange(games)
    home_side = np.where(train["neutral_site"].to_numpy() == 1, 0.0, 0.5)
    x[rows, h] = 1
    x[rows, n + a] = -1
    x[rows, 2 * n] = 1
    x[rows, 2 * n + 1] = home_side
    x[games + rows, a] = 1
    x[games + rows, n + h] = -1
    x[games + rows, 2 * n] = 1
    x[games + rows, 2 * n + 1] = -home_side
    y = np.concatenate([train["home_score"].to_numpy(float), train["away_score"].to_numpy(float)])
    penalty = np.zeros(2 * n + 2)
    penalty[:2 * n] = shrinkage
    xtw = x.T * w
    beta = np.linalg.solve(xtw @ x + np.diag(penalty), xtw @ y)
    offense, defense = beta[:n] - beta[:n].mean(), beta[n:2 * n] - beta[n:2 * n].mean()
    return {"ratings": pd.DataFrame({"offense": offense, "defense": defense}, index=teams),
            "base": beta[2 * n] + beta[:n].mean() - beta[n:2 * n].mean(), "home_adv": beta[2 * n + 1]}


def predict(model: dict, home: str, away: str, neutral: bool = False, margin_sd=MARGIN_SD) -> dict:
    r = model["ratings"]
    known = lambda t, c: r.at[t, c] if t in r.index else 0.0   # a team never seen: average
    edge = 0.0 if neutral else model["home_adv"] / 2
    home_pts = model["base"] + edge + known(home, "offense") - known(away, "defense")
    away_pts = model["base"] - edge + known(away, "offense") - known(home, "defense")
    margin = home_pts - away_pts
    return {"home_points": home_pts, "away_points": away_pts, "margin": margin, "total": home_pts + away_pts,
            "home_win": float(norm.cdf(margin / margin_sd))}


def window(played: pd.DataFrame, cutoff: str, training_years=TRAINING_YEARS) -> pd.DataFrame:
    start = (pd.Timestamp(cutoff) - pd.DateOffset(years=training_years)).strftime("%Y-%m-%d")
    return played[(played["date"] >= start) & (played["date"] < cutoff)]


def backtest(games: pd.DataFrame, start_season: int, end_season: int | None = None, half_life_days=HALF_LIFE_DAYS,
             season_weight=SEASON_WEIGHT, shrinkage=SHRINKAGE, margin_sd=MARGIN_SD,
             training_years=TRAINING_YEARS) -> pd.DataFrame:
    """Walk-forward: for each week, fit on earlier games only, then predict that week's games
    that involve at least one top-division (FBS) team."""
    played = games[games["status"] == "final"]
    test = played[(played["season"] >= start_season) & ((played["home_division"] == "fbs") | (played["away_division"] == "fbs"))]
    if end_season is not None:
        test = test[test["season"] <= end_season]
    rows = []
    for (season, _, week), week_games in test.groupby(["season", "season_type", "week"], sort=False):
        cutoff = week_games["date"].min()
        train = window(played, cutoff, training_years)
        if train.empty:
            continue
        model = fit(train, cutoff, season, half_life_days, season_weight, shrinkage)
        for g in week_games.itertuples():
            p = predict(model, g.home, g.away, bool(g.neutral_site), margin_sd)
            rows.append({"id": g.id, "model_home": p["home_win"], "model_margin": p["margin"],
                         "home_points": p["home_points"], "away_points": p["away_points"]})
    return pd.DataFrame(rows)


def elo_probability(home_elo, away_elo, neutral, home_field: float = 55.0):
    """CollegeFootballData's pre-game Elo as a win probability (a benchmark), with a home bonus."""
    diff = np.asarray(home_elo, float) - np.asarray(away_elo, float) + np.where(np.asarray(neutral) == 1, 0, home_field)
    return 1 / (1 + 10 ** (-diff / 400))
