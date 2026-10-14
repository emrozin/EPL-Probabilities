"""Elo ratings for the NFL: the baseline model.

Every team has one rating (1500 = average). Before each game:

    home win probability = 1 / (1 + 10 ** (-(home rating + home advantage - away rating) / 400))

After the game, the winner takes rating points from the loser: more for an upset, and more
for a bigger win (the margin-of-victory multiplier, damped when the favorite wins big so
strong teams' ratings don't run away). Between seasons, each rating is pulled partway back
toward 1500, because rosters change.

Home advantage can be learned from results as games are played (it has shrunk from about
2.7 points in the 2000s to under 2 since 2020), though tuning found a fixed value worked as
well. Neutral-site games (international games, the Super Bowl) get no home advantage.

This is the approach FiveThirtyEight used for its NFL forecasts; the settings here are
tuned on 2010-2017 (see nfl/tune_elo.py).
"""

import math

import numpy as np
import pandas as pd

MEAN = 1500
# Tuned on 2010-2017 (nfl/tune_elo.py)
K = 25               # how fast ratings move
HOME_ADVANTAGE = 55  # home advantage in Elo points (25 Elo points is about 1 point of margin)
HOME_LEARNING = 0.0  # how fast home advantage adapts to results, as a share of K (tuning chose 0)
REVERT = 0.5         # share of the way back to 1500 each offseason


def load_nfl_games(conn) -> pd.DataFrame:
    """Every NFL game (played and scheduled) in date order, with what Elo and the backtest need."""
    return pd.read_sql("""
        SELECT g.id, g.season, g.date, g.status, h.name AS home, a.name AS away,
               g.home_score, g.away_score, i.neutral_site, i.game_type, i.week, i.spread_line,
               i.home_qb, i.away_qb
        FROM games g
        JOIN leagues l ON l.id = g.league_id AND l.code = 'NFL'
        JOIN teams h ON h.id = g.home_team_id
        JOIN teams a ON a.id = g.away_team_id
        JOIN nfl_game_info i ON i.game_id = g.id
        ORDER BY g.date, g.id
    """, conn)


def load_nfl_market(conn) -> pd.DataFrame:
    """Market home win probability from the moneylines, with the bookmaker's margin removed."""
    df = pd.read_sql("""
        SELECT h.game_id AS id, 1 / h.price AS h, 1 / a.price AS a
        FROM odds h
        JOIN odds a ON a.game_id = h.game_id AND a.market = 'moneyline' AND a.outcome = 'away'
        WHERE h.market = 'moneyline' AND h.outcome = 'home'
    """, conn)
    df["market_home"] = df["h"] / (df["h"] + df["a"])
    return df[["id", "market_home"]]


def win_probability(rating_diff):
    """Home win probability from (home rating + home advantage - away rating)."""
    return 1 / (1 + 10 ** (-rating_diff / 400))


def margin_multiplier(margin: int, rating_diff: float) -> float:
    """Bigger wins move ratings more, with diminishing returns; a favorite's big win counts for less."""
    return math.log(abs(margin) + 1) * 2.2 / (0.001 * rating_diff + 2.2)


def run_elo(games: pd.DataFrame, k=K, home_advantage=HOME_ADVANTAGE, home_learning=HOME_LEARNING,
            revert=REVERT, use_margin=True) -> tuple[pd.DataFrame, dict, float]:
    """Play through every game in order. Returns (each game's pre-game probability, final ratings,
    final home advantage).

    Each game's probability uses only results from before it, so the output is already a fair
    walk-forward prediction. Scheduled games get a probability but don't update anything.
    """
    ratings: dict[str, float] = {}
    hfa = home_advantage
    season = None
    rows = []
    for g in games.itertuples(index=False):
        if g.season != season:
            if season is not None:  # offseason: everyone moves part of the way back to average
                ratings = {t: MEAN + (1 - revert) * (r - MEAN) for t, r in ratings.items()}
            season = g.season
        home, away = ratings.get(g.home, MEAN), ratings.get(g.away, MEAN)
        edge = 0.0 if g.neutral_site else hfa
        diff = home + edge - away
        p = win_probability(diff)
        rows.append({"id": g.id, "elo_home": p, "home_rating": home, "away_rating": away, "home_edge": edge})
        if g.status != "final":
            continue
        margin = g.home_score - g.away_score
        actual = 1.0 if margin > 0 else 0.5 if margin == 0 else 0.0
        # favorite-relative rating difference, for damping a favorite's blowout
        winner_diff = diff if margin > 0 else -diff
        step = k * (margin_multiplier(margin, winner_diff) if use_margin and margin != 0 else 1.0) * (actual - p)
        ratings[g.home] = home + step
        ratings[g.away] = away - step
        if not g.neutral_site:
            hfa += home_learning * k * (actual - p)
    return pd.DataFrame(rows), ratings, hfa


def score(df: pd.DataFrame, prob_column: str, result_column: str = "home_win") -> pd.Series:
    """Log loss per game for a home-win probability (ties excluded before calling)."""
    p = df[prob_column].clip(1e-9, 1 - 1e-9)
    y = df[result_column]
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))
