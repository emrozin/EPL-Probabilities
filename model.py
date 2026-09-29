"""Poisson match-outcome model with an xG blend.

Used by the notebook and the website:
    load_games(conn)     finished matches, with Understat xG where available
    load_market(conn)    market probabilities (closing or pre-match) with the margin removed
    fit_poisson(...)     fit team attack/defense ratings
    predict(...)         home/draw/away probabilities for one match
    backtest(...)        walk-forward backtest
    add_scores(...)      log loss and Brier score columns

Default settings were tuned on 2016/17-2020/21 and tested on 2021/22 onward.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

# Tuned settings
HALF_LIFE_DAYS = 180   # a match's weight halves every 180 days
SHRINKAGE = 0.5        # pull toward average ratings (L2 regularization)
XG_WEIGHT = 0.5        # learn from 50% actual goals, 50% xG
RHO = 0.0              # Dixon-Coles low-score correction (tested; didn't help)
TRAINING_YEARS = 3     # how far back the backtest trains


def load_games(conn, league="EPL"):
    """All finished matches for a league, oldest first."""
    return pd.read_sql("""
        SELECT g.id, g.season, g.date,
               h.name AS home, a.name AS away,
               g.home_score, g.away_score,
               x.home_xg, x.away_xg
        FROM games g
        JOIN leagues l ON l.id = g.league_id
        JOIN teams h   ON h.id = g.home_team_id
        JOIN teams a   ON a.id = g.away_team_id
        LEFT JOIN game_xg x ON x.game_id = g.id AND x.source = 'understat'
        WHERE l.code = ? AND g.status = 'final'
        ORDER BY g.date
    """, conn, params=(league,))


def load_market(conn, closing=True):
    """1X2 probabilities per game: Pinnacle, else Betfair Exchange, margin removed.

    closing=True uses closing odds (for backtests); closing=False uses pre-match odds
    (for upcoming games, whose closing odds don't exist yet).
    """
    return pd.read_sql("""
        WITH closing AS (
            SELECT game_id, bookmaker,
                   MAX(CASE WHEN outcome = 'home' THEN 1.0 / price END) AS h,
                   MAX(CASE WHEN outcome = 'draw' THEN 1.0 / price END) AS d,
                   MAX(CASE WHEN outcome = 'away' THEN 1.0 / price END) AS a
            FROM odds
            WHERE market = '1x2' AND is_closing = ?
              AND bookmaker IN ('Pinnacle', 'Betfair Exchange')
            GROUP BY game_id, bookmaker
            HAVING COUNT(*) = 3
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                          PARTITION BY game_id
                          ORDER BY bookmaker = 'Pinnacle' DESC) AS rn
            FROM closing
        )
        SELECT game_id AS id, bookmaker AS benchmark,
               h / (h + d + a) AS mkt_home,
               d / (h + d + a) AS mkt_draw,
               a / (h + d + a) AS mkt_away
        FROM ranked
        WHERE rn = 1
    """, conn, params=(1 if closing else 0,))


def fit_poisson(train, as_of, half_life_days=HALF_LIFE_DAYS, shrinkage=SHRINKAGE, xg_weight=XG_WEIGHT):
    """Fit attack/defense ratings, base rate and home advantage by weighted maximum likelihood."""
    teams = sorted(set(train["home"]) | set(train["away"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)

    home_idx = train["home"].map(index).to_numpy()
    away_idx = train["away"].map(index).to_numpy()

    # Blend actual goals with xG. Matches without xG (before 2014/15) fall back to goals.
    # astype(float) keeps the columns numeric even when no match in the window has xG.
    home_xg = train["home_xg"].astype(float).fillna(train["home_score"])
    away_xg = train["away_xg"].astype(float).fillna(train["away_score"])
    home_goals = ((1 - xg_weight) * train["home_score"] + xg_weight * home_xg).to_numpy(dtype=float)
    away_goals = ((1 - xg_weight) * train["away_score"] + xg_weight * away_xg).to_numpy(dtype=float)

    days_ago = (pd.Timestamp(as_of) - pd.to_datetime(train["date"])).dt.days.to_numpy()
    weights = 0.5 ** (days_ago / half_life_days)

    def unpack(params):
        attack = params[:n] - params[:n].mean()
        defense = params[n:2 * n] - params[n:2 * n].mean()
        return attack, defense, params[2 * n], params[2 * n + 1]

    def objective(params):
        attack, defense, base, home_adv = unpack(params)
        log_home = base + home_adv + attack[home_idx] - defense[away_idx]
        log_away = base + attack[away_idx] - defense[home_idx]
        exp_home, exp_away = np.exp(log_home), np.exp(log_away)

        # Poisson log-likelihood, minus a constant that doesn't depend on the ratings
        log_lik = home_goals * log_home - exp_home + away_goals * log_away - exp_away
        value = -np.sum(weights * log_lik) + shrinkage * np.sum(attack**2 + defense**2)

        # Gradient: how the value changes as each rating changes
        res_home = weights * (exp_home - home_goals)
        res_away = weights * (exp_away - away_goals)
        grad_attack = (np.bincount(home_idx, res_home, n) + np.bincount(away_idx, res_away, n)
                       + 2 * shrinkage * attack)
        grad_defense = (-np.bincount(away_idx, res_home, n) - np.bincount(home_idx, res_away, n)
                        + 2 * shrinkage * defense)
        grad = np.concatenate([
            grad_attack - grad_attack.mean(),    # account for the centering in unpack()
            grad_defense - grad_defense.mean(),
            [res_home.sum() + res_away.sum(), res_home.sum()],
        ])
        return value, grad

    result = minimize(objective, np.zeros(2 * n + 2), jac=True, method="L-BFGS-B")
    if not result.success:
        print("Warning: optimizer did not converge:", result.message)

    attack, defense, base, home_adv = unpack(result.x)
    return {
        "ratings": pd.DataFrame({"attack": np.exp(attack), "defense": np.exp(defense)}, index=teams),
        "base": np.exp(base),
        "home_adv": np.exp(home_adv),
    }


def predict(model, home, away, max_goals=10, rho=RHO):
    """Home/draw/away probabilities, expected goals, and the full scoreline grid."""
    r = model["ratings"]
    exp_home = model["base"] * model["home_adv"] * r.loc[home, "attack"] / r.loc[away, "defense"]
    exp_away = model["base"] * r.loc[away, "attack"] / r.loc[home, "defense"]

    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    # Dixon-Coles adjustment: more 0-0 and 1-1, fewer 1-0 and 0-1 (when rho < 0)
    grid[0, 0] *= 1 - exp_home * exp_away * rho
    grid[0, 1] *= 1 + exp_home * rho
    grid[1, 0] *= 1 + exp_away * rho
    grid[1, 1] *= 1 - rho

    return {
        "exp_home": exp_home,
        "exp_away": exp_away,
        "home": np.tril(grid, -1).sum(),
        "draw": np.trace(grid),
        "away": np.triu(grid, 1).sum(),
        "grid": grid,
    }


def backtest(games, start_season, rho=RHO, **model_settings):
    """Walk-forward: each week, train only on earlier matches, then predict that week's games."""
    test = games[games["season"] >= start_season].copy()
    test["week"] = pd.to_datetime(test["date"]).dt.to_period("W").dt.start_time

    rows = []
    for week_start, week_games in test.groupby("week"):
        cutoff = week_start.strftime("%Y-%m-%d")
        window_start = (week_start - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
        train = games[(games["date"] >= window_start) & (games["date"] < cutoff)]
        model = fit_poisson(train, cutoff, **model_settings)
        known = set(model["ratings"].index)

        for g in week_games.itertuples():
            if g.home not in known or g.away not in known:
                continue  # a team with no games in the training window yet
            p = predict(model, g.home, g.away, rho=rho)
            if g.home_score > g.away_score:
                result = "home"
            elif g.home_score == g.away_score:
                result = "draw"
            else:
                result = "away"
            rows.append({"id": g.id, "season": g.season, "result": result,
                         "model_home": p["home"], "model_draw": p["draw"], "model_away": p["away"]})
    return pd.DataFrame(rows)


def add_scores(df, prefix):
    """Add <prefix>_logloss and <prefix>_brier columns, from <prefix>_home/_draw/_away."""
    outcomes = ["home", "draw", "away"]
    probs = df[[f"{prefix}_{o}" for o in outcomes]].to_numpy()
    actual = (df["result"].to_numpy()[:, None] == np.array(outcomes)).astype(float)
    df[f"{prefix}_logloss"] = -np.log((probs * actual).sum(axis=1))
    df[f"{prefix}_brier"] = ((probs - actual) ** 2).sum(axis=1)
