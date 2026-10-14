"""NFL Elo: valid probabilities, ratings that move the right way, and no look-ahead."""

import pandas as pd
import pytest

from nfl.elo import MEAN, margin_multiplier, run_elo, win_probability


def game(id, season, date, home, away, home_score=None, away_score=None, neutral=0):
    return {"id": id, "season": season, "date": date, "status": "final" if home_score is not None else "scheduled",
            "home": home, "away": away, "home_score": home_score, "away_score": away_score,
            "neutral_site": neutral, "game_type": "REG", "week": 1, "spread_line": None}


def frame(*games):
    return pd.DataFrame(list(games))


def test_win_probability_is_symmetric():
    assert win_probability(0) == pytest.approx(0.5)
    assert win_probability(100) + win_probability(-100) == pytest.approx(1.0)
    assert win_probability(400) == pytest.approx(10 / 11)


def test_bigger_wins_count_more_but_a_favorites_blowout_is_damped():
    assert margin_multiplier(21, 0) > margin_multiplier(3, 0)
    assert margin_multiplier(21, 200) < margin_multiplier(21, 0)


def test_winner_gains_what_the_loser_loses_and_home_advantage_applies():
    predictions, ratings, _ = run_elo(frame(game(1, 2020, "2020-09-10", "KC", "HOU", 34, 20)), home_advantage=55)
    assert predictions.loc[0, "elo_home"] == pytest.approx(win_probability(55))
    assert ratings["KC"] > MEAN > ratings["HOU"]
    assert ratings["KC"] + ratings["HOU"] == pytest.approx(2 * MEAN)


def test_neutral_site_games_have_no_home_advantage():
    predictions, _, _ = run_elo(frame(game(1, 2020, "2020-10-01", "JAX", "ATL", 20, 17, neutral=1)), home_advantage=55)
    assert predictions.loc[0, "elo_home"] == pytest.approx(0.5)


def test_ratings_move_back_toward_average_between_seasons():
    games = frame(game(1, 2020, "2020-09-10", "KC", "HOU", 40, 0), game(2, 2021, "2021-09-10", "KC", "HOU"))
    predictions, _, _ = run_elo(games, revert=0.5)
    end_of_2020 = run_elo(games.iloc[:1], revert=0.5)[1]["KC"]
    assert predictions.loc[1, "home_rating"] == pytest.approx(MEAN + 0.5 * (end_of_2020 - MEAN))


def test_predictions_never_use_later_results():
    first = game(1, 2020, "2020-09-10", "KC", "HOU", 30, 10)
    second_a = game(2, 2020, "2020-09-17", "HOU", "KC", 10, 30)
    second_b = game(2, 2020, "2020-09-17", "HOU", "KC", 30, 10)
    third = game(3, 2020, "2020-09-24", "KC", "HOU")
    a, _, _ = run_elo(frame(first, second_a, third))
    b, _, _ = run_elo(frame(first, second_b, third))
    assert a.loc[1, "elo_home"] == b.loc[1, "elo_home"]       # game 2's own result doesn't affect its prediction
    assert a.loc[2, "elo_home"] != b.loc[2, "elo_home"]       # but it does affect later games


def test_scheduled_games_get_a_probability_but_change_nothing():
    games = frame(game(1, 2020, "2020-09-10", "KC", "HOU", 30, 10), game(2, 2020, "2020-09-17", "HOU", "KC"))
    predictions, ratings, _ = run_elo(games)
    assert 0 < predictions.loc[1, "elo_home"] < 0.5          # KC, now rated higher, favored away from home
    assert ratings == run_elo(games.iloc[:1])[1]


def test_ties_leave_equal_teams_unchanged():
    _, ratings, _ = run_elo(frame(game(1, 2020, "2020-09-10", "NYG", "WAS", 20, 20, neutral=1)))
    assert ratings["NYG"] == pytest.approx(MEAN) and ratings["WAS"] == pytest.approx(MEAN)
