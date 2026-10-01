"""Poisson match-outcome model with an xG blend.

Used by the notebook and the website:
    load_games(conn)     finished matches, with Understat xG where available
    load_market(conn)    market probabilities (closing or pre-match) with the margin removed
    load_totals(conn)    market probability of over 2.5 goals, margin removed
    scoreline_grid(...)  probability of every exact score, from each side's expected goals
    goal_markets(grid)   home/draw/away, over/under, both teams to score, likeliest scores
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


def load_totals(conn, closing=True, line=2.5):
    """Over/under probability per game: Pinnacle, else Betfair Exchange, margin removed."""
    return pd.read_sql("""
        WITH totals AS (
            SELECT game_id, bookmaker,
                   MAX(CASE WHEN outcome = 'over'  THEN 1.0 / price END) AS o,
                   MAX(CASE WHEN outcome = 'under' THEN 1.0 / price END) AS u
            FROM odds
            WHERE market = 'total' AND line = ? AND is_closing = ?
              AND bookmaker IN ('Pinnacle', 'Betfair Exchange')
            GROUP BY game_id, bookmaker
            HAVING COUNT(*) = 2
        ),
        ranked AS (
            SELECT *, ROW_NUMBER() OVER (
                          PARTITION BY game_id
                          ORDER BY bookmaker = 'Pinnacle' DESC) AS rn
            FROM totals
        )
        SELECT game_id AS id, bookmaker AS totals_benchmark, o / (o + u) AS mkt_over
        FROM ranked
        WHERE rn = 1
    """, conn, params=(line, 1 if closing else 0))


def squad_value_prior(teams, squad_values, value_weight):
    """Where shrinkage should pull each team's ratings: toward what its squad value suggests.

    squad_values: {team: value} for this week. Each team's log value is compared with the average
    of the teams being rated, so doubling a squad's value counts the same at any level. A team
    with no value gets 0 (average), as does every team when value_weight is 0.
    """
    if not squad_values or not value_weight:
        return np.zeros(len(teams))
    logs = np.array([np.log(squad_values[t]) if squad_values.get(t, 0) > 0 else np.nan for t in teams])
    if np.isnan(logs).all():
        return np.zeros(len(teams))
    return np.nan_to_num(value_weight * (logs - np.nanmean(logs)), nan=0.0)


def fit_poisson(train, as_of, half_life_days=HALF_LIFE_DAYS, shrinkage=SHRINKAGE, xg_weight=XG_WEIGHT,
                squad_values=None, value_weight=0.0):
    """Fit attack/defense ratings, base rate and home advantage by weighted maximum likelihood.

    Shrinkage pulls ratings toward average or, with squad_values and a value_weight above 0,
    toward what each team's squad value suggests (see squad_value_prior).
    """
    teams = sorted(set(train["home"]) | set(train["away"]))
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    prior = squad_value_prior(teams, squad_values, value_weight)
    prior = prior - prior.mean()   # centred, like the ratings themselves

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
        value = -np.sum(weights * log_lik) + shrinkage * np.sum((attack - prior)**2 + (defense - prior)**2)

        # Gradient: how the value changes as each rating changes
        res_home = weights * (exp_home - home_goals)
        res_away = weights * (exp_away - away_goals)
        grad_attack = (np.bincount(home_idx, res_home, n) + np.bincount(away_idx, res_away, n)
                       + 2 * shrinkage * (attack - prior))
        grad_defense = (-np.bincount(away_idx, res_home, n) - np.bincount(home_idx, res_away, n)
                        + 2 * shrinkage * (defense - prior))
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


def scoreline_grid(exp_home, exp_away, max_goals=15, rho=RHO):
    """grid[i, j] = probability the home side scores i and the away side j."""
    goals = np.arange(max_goals + 1)
    grid = np.outer(poisson.pmf(goals, exp_home), poisson.pmf(goals, exp_away))

    # Dixon-Coles adjustment: more 0-0 and 1-1, fewer 1-0 and 0-1 (when rho < 0)
    grid[0, 0] *= 1 - exp_home * exp_away * rho
    grid[0, 1] *= 1 + exp_home * rho
    grid[1, 0] *= 1 + exp_away * rho
    grid[1, 1] *= 1 - rho

    # The grid stops at max_goals (15 per side, far beyond any real score), so rescale it to cover
    # the vanishingly small chance of more goals than that.
    return grid / grid.sum()


TOTAL_LINES = (1.5, 2.5, 3.5)


def goal_markets(grid, top=5):
    """Every market that follows from a scoreline grid."""
    n = grid.shape[0]
    total_goals = np.add.outer(np.arange(n), np.arange(n))
    markets = {
        "home": np.tril(grid, -1).sum(),
        "draw": np.trace(grid),
        "away": np.triu(grid, 1).sum(),
        "btts": grid[1:, 1:].sum(),                       # both teams score
    }
    for line in TOTAL_LINES:
        markets[f"over_{line}"] = grid[total_goals > line].sum()
        markets[f"under_{line}"] = 1 - markets[f"over_{line}"]
    order = np.argsort(grid, axis=None)[::-1][:top]
    markets["top_scores"] = [(int(i), int(j), float(grid[i, j])) for i, j in zip(*np.unravel_index(order, grid.shape))]
    return markets


def predict(model, home, away, max_goals=15, rho=RHO):
    """Expected goals, every goal market, and the full scoreline grid for one match."""
    r = model["ratings"]
    exp_home = model["base"] * model["home_adv"] * r.loc[home, "attack"] / r.loc[away, "defense"]
    exp_away = model["base"] * r.loc[away, "attack"] / r.loc[home, "defense"]
    grid = scoreline_grid(exp_home, exp_away, max_goals, rho)
    return {"exp_home": exp_home, "exp_away": exp_away, "grid": grid, **goal_markets(grid)}


def load_squad_values(conn, league="EPL") -> dict:
    """{'YYYY-MM-DD' Monday: {team: squad value in euros}}; empty if squad values haven't been built."""
    try:
        rows = conn.execute("SELECT week, team, value_eur FROM squad_values WHERE league_code = ?", (league,)).fetchall()
    except Exception:  # no squad_values table yet
        return {}
    weeks: dict = {}
    for week, team, value in rows:
        weeks.setdefault(week, {})[team] = value
    return weeks


def backtest(games, start_season, rho=RHO, squad_values_by_week=None, **model_settings):
    """Walk-forward: each week, train only on earlier matches, then predict that week's games.

    squad_values_by_week: optional {Monday: {team: value}}; each week uses that Monday's values,
    which are built only from information available by then.
    """
    test = games[games["season"] >= start_season].copy()
    test["week"] = pd.to_datetime(test["date"]).dt.to_period("W").dt.start_time

    rows = []
    for week_start, week_games in test.groupby("week"):
        cutoff = week_start.strftime("%Y-%m-%d")
        window_start = (week_start - pd.DateOffset(years=TRAINING_YEARS)).strftime("%Y-%m-%d")
        train = games[(games["date"] >= window_start) & (games["date"] < cutoff)]
        if train.empty:
            continue  # no earlier matches to learn from (e.g. the first weeks of a league's data)
        if squad_values_by_week is not None:
            model_settings["squad_values"] = squad_values_by_week.get(cutoff, {})
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
                         "model_home": p["home"], "model_draw": p["draw"], "model_away": p["away"],
                         "exp_home": p["exp_home"], "exp_away": p["exp_away"]})
    return pd.DataFrame(rows)


def add_scores(df, prefix):
    """Add <prefix>_logloss and <prefix>_brier columns, from <prefix>_home/_draw/_away."""
    outcomes = ["home", "draw", "away"]
    probs = df[[f"{prefix}_{o}" for o in outcomes]].to_numpy()
    actual = (df["result"].to_numpy()[:, None] == np.array(outcomes)).astype(float)
    df[f"{prefix}_logloss"] = -np.log((probs * actual).sum(axis=1))
    df[f"{prefix}_brier"] = ((probs - actual) ** 2).sum(axis=1)
