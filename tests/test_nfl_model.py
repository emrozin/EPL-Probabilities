"""NFL points model: recovers known strengths and the QB effect, and never looks ahead."""

import numpy as np
import pandas as pd
import pytest

from nfl.model import add_qb_changes, backtest, fit, predict


def simulate(seed=0, seasons=range(2015, 2019), qb_effect=-4.0, home_adv=2.0):
    """A 12-team league with known offense/defense and occasional backup quarterbacks."""
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    off = dict(zip(teams, rng.normal(0, 4, 12)))
    dfn = dict(zip(teams, rng.normal(0, 4, 12)))
    rows = []
    for season in seasons:
        start = pd.Timestamp(f"{season}-09-10")
        for week in range(1, 18):
            order = rng.permutation(teams)
            for k in range(6):
                h, a = order[2 * k], order[2 * k + 1]
                hb, ab = rng.random() < 0.15, rng.random() < 0.15
                hp = 22 + home_adv / 2 + off[h] - dfn[a] + qb_effect * hb + rng.normal(0, 7)
                ap = 22 - home_adv / 2 + off[a] - dfn[h] + qb_effect * ab + rng.normal(0, 7)
                rows.append({"id": len(rows) + 1, "season": season, "week": week, "status": "final",
                             "date": (start + pd.Timedelta(days=7 * week)).strftime("%Y-%m-%d"),
                             "home": h, "away": a, "home_score": round(hp), "away_score": round(ap),
                             "neutral_site": 0, "home_qb": f"{h}-backup" if hb else f"{h}-starter",
                             "away_qb": f"{a}-backup" if ab else f"{a}-starter"})
    return pd.DataFrame(rows), off, dfn


def test_usual_starter_is_based_only_on_earlier_games():
    games = pd.DataFrame([
        {"id": 1, "date": "2020-09-10", "status": "final", "home": "A", "away": "B", "home_qb": "Al", "away_qb": "Bo"},
        {"id": 2, "date": "2020-09-17", "status": "final", "home": "A", "away": "B", "home_qb": "Ann", "away_qb": "Bo"},
        {"id": 3, "date": "2020-09-24", "status": "scheduled", "home": "B", "away": "A", "home_qb": None, "away_qb": None},
    ])
    flags = add_qb_changes(games).set_index("id")
    assert flags.loc[1, "home_qb_change"] == 0      # no history yet: not counted as a change
    assert flags.loc[2, "home_qb_change"] == 1      # Ann replaces the usual starter Al
    assert flags.loc[2, "away_qb_change"] == 0
    assert flags.loc[3, ["home_qb_change", "away_qb_change"]].tolist() == [0, 0]   # starter unknown yet


def test_fit_recovers_team_strengths_home_advantage_and_qb_effect():
    games, off, dfn = simulate()
    games = add_qb_changes(games)
    model = fit(games, "2019-06-01", half_life_days=100_000, shrinkage=0.01)
    r = model["ratings"]
    assert np.corrcoef(r["offense"], pd.Series(off)[r.index])[0, 1] > 0.9
    assert np.corrcoef(r["defense"], pd.Series(dfn)[r.index])[0, 1] > 0.9
    assert model["home_adv"] == pytest.approx(2.0, abs=1.0)
    assert model["qb_change"] == pytest.approx(-4.0, abs=1.5)


def test_predictions_are_consistent():
    games, _, _ = simulate(seed=1)
    model = fit(add_qb_changes(games), "2019-06-01")
    p = predict(model, "T0", "T1")
    assert p["margin"] == pytest.approx(p["home_points"] - p["away_points"])
    assert 0 < p["home_win"] < 1
    neutral = predict(model, "T0", "T1", neutral=True)
    assert neutral["margin"] == pytest.approx(p["margin"] - model["home_adv"])
    backup = predict(model, "T0", "T1", home_qb_change=1)
    assert backup["home_win"] < p["home_win"]
    swapped = predict(model, "T1", "T0", neutral=True)
    assert swapped["home_win"] == pytest.approx(1 - neutral["home_win"])


def test_fit_works_when_no_game_has_a_qb_change():
    games, _, _ = simulate(seed=2)
    model = fit(games.assign(home_qb_change=0, away_qb_change=0), "2019-06-01")
    assert model["qb_change"] == pytest.approx(0, abs=1e-6)


def test_backtest_never_trains_on_the_week_it_predicts(monkeypatch):
    import nfl.model as m
    games, _, _ = simulate(seed=3)
    games = add_qb_changes(games)
    seen = []
    real_fit = m.fit

    def spy(train, as_of, *args):
        seen.append((train["date"].max(), as_of))
        return real_fit(train, as_of, *args)

    monkeypatch.setattr(m, "fit", spy)
    results = m.backtest(games, 2017)
    assert len(results) > 0
    assert all(last < cutoff for last, cutoff in seen)
