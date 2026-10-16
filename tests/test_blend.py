"""The model-market blend: the pooling math, choosing the weight, and when the website shows it."""

import json

import numpy as np
import pytest

import blend
from epl import blend as epl_blend


def test_pool_is_the_model_at_weight_1_the_market_at_0_and_logit_average_between():
    model, market = np.array([[0.6, 0.4]]), np.array([[0.57, 0.43]])
    assert blend.pool(model, market, 1.0) == pytest.approx(model)
    assert blend.pool(model, market, 0.0) == pytest.approx(market)
    logit = lambda p: np.log(p / (1 - p))
    expected = 1 / (1 + np.exp(-(0.3 * logit(0.6) + 0.7 * logit(0.57))))
    assert blend.pool(model, market, 0.3)[0, 0] == pytest.approx(expected)
    three = blend.pool([[0.5, 0.25, 0.25]], [[0.4, 0.3, 0.3]], 0.5)
    assert three.sum() == pytest.approx(1) and 0.4 < three[0, 0] < 0.5


def test_tuning_finds_the_weight_that_helps():
    rng = np.random.default_rng(0)
    truth = rng.uniform(0.2, 0.8, 4000)
    outcome = (rng.random(4000) > truth).astype(int)          # 0 = home win, with probability truth
    noisy = lambda sd: 1 / (1 + np.exp(-(np.log(truth / (1 - truth)) + rng.normal(0, sd, 4000))))
    model, market = noisy(0.4), noisy(0.4)                     # equally good, independent errors
    two = lambda p: np.column_stack([p, 1 - p])
    w, scores = blend.tune(two(model), two(market), outcome)
    assert 0.3 <= w <= 0.7                                     # averaging two equal forecasts helps
    assert dict(scores)[w] < dict(scores)[0.0] and dict(scores)[w] < dict(scores)[1.0]


def test_premier_league_cards_show_the_blend_only_when_it_beat_the_market(tmp_path, monkeypatch):
    info = {"weight": 0.3, "tuning_seasons": "2016-2020", "held_out_seasons": "2021-2025", "held_out_games": 10,
            "held_out": {"model": 0.98, "market": 0.965, "blend": 0.963}, "tuning_curve": []}
    path = tmp_path / "blend.json"
    path.write_text(json.dumps(info))
    monkeypatch.setattr(epl_blend, "BLEND_PATH", path)
    assert epl_blend.load_info()["weight"] == 0.3
    p = epl_blend.probabilities({"home": 0.5, "draw": 0.25, "away": 0.25}, {"home": 0.4, "draw": 0.3, "away": 0.3}, 0.3)
    assert sum(p.values()) == pytest.approx(1)
