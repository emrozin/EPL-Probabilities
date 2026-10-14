"""Season simulator: play out the rest of the season thousands of times with the model.

For every match still to be played, each simulated season draws a score from the model's
expected goals (independent Poisson draws, as in the model itself), adds it to the current
table, and ranks the teams by points, then goal difference, then goals scored. Counting
where each team finishes across all simulations gives its chances of every position.

Remaining fixtures don't need a separate data source: every team plays every other team
once at home and once away, so they're all the home/away pairings not yet played.

Caveat: team ratings are held fixed for the rest of the season, so the simulation doesn't
account for them changing (injuries, transfers, form). Real seasons are somewhat less
predictable than it suggests, especially early on.
"""

import numpy as np
import pandas as pd

SIMULATIONS = 10_000
SEED = 2026  # fixed, so the page shows the same numbers every time it's built from the same data


def remaining_fixtures(season_games: pd.DataFrame) -> list[tuple[str, str]]:
    """Every home/away pairing among this season's teams that hasn't been played yet."""
    teams = sorted(set(season_games["home"]) | set(season_games["away"]))
    played = set(zip(season_games["home"], season_games["away"]))
    return [(h, a) for h in teams for a in teams if h != a and (h, a) not in played]


def simulate_season(table: pd.DataFrame, fixtures: list[tuple[str, str]], expected_goals,
                    simulations: int = SIMULATIONS, seed: int = SEED) -> dict:
    """
    table: current standings with columns team, points, gf, ga (one row per team).
    expected_goals(home, away) -> (exp_home, exp_away) from the model.

    Returns each team's probability of every finishing position, expected points,
    and the matrix of positions (simulations x teams) for any further summaries.
    """
    rng = np.random.default_rng(seed)
    teams = list(table["team"])
    index = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)

    points = np.tile(table["points"].to_numpy(float), (simulations, 1))
    gf = np.tile(table["gf"].to_numpy(float), (simulations, 1))
    ga = np.tile(table["ga"].to_numpy(float), (simulations, 1))

    if fixtures:
        lam = np.array([expected_goals(h, a) for h, a in fixtures])          # (fixtures, 2)
        home_goals = rng.poisson(lam[:, 0], size=(simulations, len(fixtures)))
        away_goals = rng.poisson(lam[:, 1], size=(simulations, len(fixtures)))
        home_idx = np.array([index[h] for h, _ in fixtures])
        away_idx = np.array([index[a] for _, a in fixtures])
        home_pts = np.where(home_goals > away_goals, 3, np.where(home_goals == away_goals, 1, 0))
        away_pts = np.where(away_goals > home_goals, 3, np.where(home_goals == away_goals, 1, 0))
        # Add every simulated match to the right team's column, for all simulations at once.
        for goals_for, goals_against, pts, idx in ((home_goals, away_goals, home_pts, home_idx),
                                                   (away_goals, home_goals, away_pts, away_idx)):
            np.add.at(points.T, idx, pts.T)
            np.add.at(gf.T, idx, goals_for.T)
            np.add.at(ga.T, idx, goals_against.T)

    # Rank by points, then goal difference, then goals scored; any remaining tie is broken at random.
    tiebreak = rng.random((simulations, n_teams))
    key = points * 1e6 + (gf - ga + 500) * 1e3 + gf + tiebreak
    order = np.argsort(-key, axis=1)                  # teams from 1st to last, per simulation
    positions = np.empty_like(order)
    positions[np.arange(simulations)[:, None], order] = np.arange(1, n_teams + 1)

    position_probs = np.stack([(positions == p).mean(axis=0) for p in range(1, n_teams + 1)], axis=1)
    return {
        "teams": teams,
        "position_probs": position_probs,            # (teams, positions)
        "expected_points": points.mean(axis=0),
        "positions": positions,
        "remaining": len(fixtures),
        "simulations": simulations,
    }
