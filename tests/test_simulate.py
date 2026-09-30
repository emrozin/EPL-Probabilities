"""The season simulator."""

import numpy as np
import pandas as pd
import pytest

from simulate import remaining_fixtures, simulate_season


def test_remaining_fixtures_are_the_unplayed_home_and_away_pairings():
    played = pd.DataFrame({"home": ["A", "B", "C"], "away": ["B", "C", "A"]})
    remaining = remaining_fixtures(played)
    assert len(remaining) == 3 * 2 - 3
    assert set(remaining) == {("B", "A"), ("C", "B"), ("A", "C")}


def test_a_finished_season_is_certain():
    table = pd.DataFrame({"team": ["A", "B", "C"], "points": [9, 4, 1], "gf": [8, 5, 2], "ga": [2, 5, 8]})
    sim = simulate_season(table, [], lambda h, a: (1.0, 1.0), simulations=200)
    assert sim["position_probs"][:, 0].tolist() == [1.0, 0.0, 0.0]
    assert sim["expected_points"].tolist() == [9, 4, 1]


def test_probabilities_are_valid_and_points_add_up():
    teams = [f"T{i}" for i in range(6)]
    table = pd.DataFrame({"team": teams, "points": 0, "gf": 0, "ga": 0})
    fixtures = [(h, a) for h in teams for a in teams if h != a]
    sim = simulate_season(table, fixtures, lambda h, a: (1.4, 1.1), simulations=4000)
    probs = sim["position_probs"]
    assert np.allclose(probs.sum(axis=0), 1)   # every position is filled once per season
    assert np.allclose(probs.sum(axis=1), 1)   # every team finishes somewhere
    # Each simulated match hands out 3 points (a win) or 2 (a draw).
    total = sim["expected_points"].sum()
    assert 2 * len(fixtures) < total < 3 * len(fixtures)


def test_stronger_teams_finish_higher():
    teams = ["Strong", "Middle", "Weak"]
    strength = {"Strong": 1.0, "Middle": 0.0, "Weak": -1.0}
    table = pd.DataFrame({"team": teams, "points": 0, "gf": 0, "ga": 0})
    fixtures = [(h, a) for h in teams for a in teams if h != a] * 5
    sim = simulate_season(table, fixtures,
                          lambda h, a: (np.exp(0.3 + strength[h] - strength[a]), np.exp(0.2 + strength[a] - strength[h])),
                          simulations=2000)
    title = dict(zip(sim["teams"], sim["position_probs"][:, 0]))
    assert title["Strong"] > 0.9 and title["Weak"] < 0.01


def test_same_seed_gives_the_same_forecast():
    teams = ["A", "B", "C", "D"]
    table = pd.DataFrame({"team": teams, "points": [3, 1, 1, 0], "gf": [2, 1, 1, 0], "ga": [0, 1, 1, 2]})
    fixtures = [("A", "C"), ("B", "D"), ("C", "D")]
    one = simulate_season(table, fixtures, lambda h, a: (1.3, 1.0), simulations=500)
    two = simulate_season(table, fixtures, lambda h, a: (1.3, 1.0), simulations=500)
    assert np.array_equal(one["position_probs"], two["position_probs"])
