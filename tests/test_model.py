"""The model: valid probabilities, sensible ratings, and a backtest that never sees the future."""

import numpy as np
import pandas as pd
import pytest

import model
from conftest import simulate_league
from model import add_scores, backtest, fit_poisson, predict


@pytest.fixture(scope="module")
def fitted():
    games, truth = simulate_league(seed=1)
    return fit_poisson(games, games["date"].max()), truth


def test_probabilities_are_valid_and_sum_to_one(fitted):
    fit, _ = fitted
    teams = list(fit["ratings"].index)
    for home, away in [(teams[0], teams[1]), (teams[2], teams[5]), (teams[9], teams[3])]:
        p = predict(fit, home, away)
        assert p["home"] + p["draw"] + p["away"] == pytest.approx(1.0, abs=1e-6)
        assert all(0 < p[o] < 1 for o in ("home", "draw", "away"))


def test_stronger_team_is_favoured(fitted):
    fit, _ = fitted
    strength = np.log(fit["ratings"]["attack"]) + np.log(fit["ratings"]["defense"])
    best, worst = strength.idxmax(), strength.idxmin()
    assert predict(fit, best, worst)["home"] > 0.5
    assert predict(fit, worst, best)["away"] > predict(fit, worst, best)["home"]


def test_fit_recovers_known_team_strengths():
    games, truth = simulate_league(seed=2, n_teams=16, seasons=range(2010, 2020))
    fit = fit_poisson(games, games["date"].max(), half_life_days=100_000, shrinkage=0.0, xg_weight=0.0)
    ratings = fit["ratings"]
    true_attack = pd.Series(truth["attack"])[ratings.index]
    true_defense = pd.Series(truth["defense"])[ratings.index]
    assert np.corrcoef(np.log(ratings["attack"]), true_attack)[0, 1] > 0.9
    assert np.corrcoef(np.log(ratings["defense"]), true_defense)[0, 1] > 0.9
    assert np.log(fit["home_adv"]) == pytest.approx(truth["home_boost"], abs=0.08)


def test_ratings_are_centred_on_an_average_team(fitted):
    fit, _ = fitted
    assert np.exp(np.log(fit["ratings"]["attack"]).mean()) == pytest.approx(1.0, abs=1e-6)
    assert np.exp(np.log(fit["ratings"]["defense"]).mean()) == pytest.approx(1.0, abs=1e-6)


def test_missing_xg_falls_back_to_goals():
    """Regression: with no xG at all, the columns arrive as objects rather than numbers."""
    games, _ = simulate_league(seed=3, seasons=range(2010, 2012))
    no_xg = games.assign(home_xg=None, away_xg=None).astype({"home_xg": object, "away_xg": object})
    as_of = games["date"].max()
    blended = fit_poisson(no_xg, as_of, xg_weight=0.5)
    goals_only = fit_poisson(games.assign(home_xg=np.nan, away_xg=np.nan), as_of, xg_weight=0.0)
    pd.testing.assert_frame_equal(blended["ratings"], goals_only["ratings"], atol=1e-6)


def test_backtest_never_trains_on_matches_from_the_week_it_predicts(monkeypatch):
    games, _ = simulate_league(seed=4, seasons=range(2014, 2018))
    real_fit = model.fit_poisson
    calls = []

    def recording_fit(train, as_of, **settings):
        calls.append((train["date"].max(), as_of))
        return real_fit(train, as_of, **settings)

    monkeypatch.setattr(model, "fit_poisson", recording_fit)
    results = backtest(games, start_season=2016)

    assert calls and len(results) > 0
    for latest_training_match, cutoff in calls:
        assert latest_training_match < cutoff

    # ...and every predicted match is played on or after the cutoff it was predicted from.
    dates = games.set_index("id")["date"]
    week_start = pd.to_datetime(dates).dt.to_period("W").dt.start_time.dt.strftime("%Y-%m-%d")
    assert (dates[results["id"]] >= week_start[results["id"]]).all()
    assert set(week_start[results["id"]]) <= {cutoff for _, cutoff in calls}


def test_scores_for_perfect_and_uninformed_forecasts():
    df = pd.DataFrame({"result": ["home", "draw", "away"],
                       "a_home": [1.0, 0.0, 0.0], "a_draw": [0.0, 1.0, 0.0], "a_away": [0.0, 0.0, 1.0],
                       "b_home": [1 / 3] * 3, "b_draw": [1 / 3] * 3, "b_away": [1 / 3] * 3})
    add_scores(df, "a")
    add_scores(df, "b")
    assert df["a_logloss"].max() == pytest.approx(0.0)
    assert df["a_brier"].max() == pytest.approx(0.0)
    assert df["b_logloss"].mean() == pytest.approx(np.log(3))


def test_goal_markets_are_consistent():
    from scipy.stats import poisson
    from model import goal_markets, scoreline_grid

    for exp_home, exp_away in [(1.4, 1.2), (2.8, 0.6), (0.7, 0.9)]:
        grid = scoreline_grid(exp_home, exp_away)
        m = goal_markets(grid)
        assert m["home"] + m["draw"] + m["away"] == pytest.approx(1.0)
        for line in (1.5, 2.5, 3.5):
            assert m[f"over_{line}"] + m[f"under_{line}"] == pytest.approx(1.0)
        assert m["over_1.5"] > m["over_2.5"] > m["over_3.5"]
        # With no draw adjustment, total goals follow a Poisson distribution with the combined average.
        assert m["over_2.5"] == pytest.approx(1 - poisson.cdf(2, exp_home + exp_away), abs=1e-5)
        scores = [p for _, _, p in m["top_scores"]]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == pytest.approx(grid.max())


def test_predict_includes_goal_markets(fitted):
    fit, _ = fitted
    teams = list(fit["ratings"].index)
    p = predict(fit, teams[0], teams[1])
    assert {"over_2.5", "under_2.5", "btts", "top_scores", "exp_home", "exp_away"} <= set(p)


def test_backtest_skips_weeks_with_no_earlier_matches():
    """Regression: a league whose data starts after the backtest's first season used to crash."""
    games, _ = simulate_league(seed=6, seasons=range(2022, 2024))
    results = backtest(games, start_season=2016)
    assert len(results) > 0
    assert results["id"].isin(games.loc[games["season"] == 2022, "id"]).any()  # 2022 predicted once data exists
