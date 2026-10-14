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


def test_quarterback_columns_only_look_backward():
    games = pd.DataFrame([
        {"id": 1, "date": "2020-09-10", "status": "final", "home": "A", "away": "B", "home_qb": "Al", "away_qb": "Bo"},
        {"id": 2, "date": "2020-09-17", "status": "final", "home": "A", "away": "B", "home_qb": "Ann", "away_qb": "Bo"},
        {"id": 3, "date": "2020-09-24", "status": "scheduled", "home": "B", "away": "A", "home_qb": None, "away_qb": float("nan")},
    ])
    q = add_qb_changes(games).set_index("id")
    assert q.loc[1, "home_qb_change"] == 0 and q.loc[1, "home_qb_starts"] == 0   # no history yet
    assert q.loc[2, "home_qb_change"] == 1 and q.loc[2, "home_qb_starts"] == 0   # Ann replaces the usual starter
    assert q.loc[2, "away_qb_starts"] == 1                                        # Bo's second start
    # Not played yet, no starters listed: the usual starters are assumed, with their starts so far.
    assert q.loc[3, "home_qb_expected"] == "Bo" and q.loc[3, "home_qb_starts"] == 2
    assert q.loc[3, ["home_qb_change", "away_qb_change"]].tolist() == [0, 0]


def test_fit_recovers_team_strengths_home_advantage_and_backup_quarterbacks():
    games, off, dfn = simulate()
    games = add_qb_changes(games)
    model = fit(games, "2019-06-01", half_life_days=100_000, shrinkage=0.01, qb_shrinkage=0.5)
    r = model["ratings"]
    starters = [model["qb_ratings"][f"{t}-starter"] for t in r.index]
    offense = r["offense"] + np.array(starters) - np.mean(starters)   # team + its usual quarterback
    assert np.corrcoef(offense, pd.Series(off)[r.index])[0, 1] > 0.9
    assert np.corrcoef(r["defense"], pd.Series(dfn)[r.index])[0, 1] > 0.9
    assert model["home_adv"] == pytest.approx(2.0, abs=1.0)
    gaps = [model["qb_ratings"][f"{t}-backup"] - model["qb_ratings"][f"{t}-starter"] for t in r.index]
    assert np.mean(gaps) == pytest.approx(-4.0, abs=1.5)    # backups cost about the simulated 4 points


def test_predictions_are_consistent():
    games, _, _ = simulate(seed=1)
    model = fit(add_qb_changes(games), "2019-06-01")
    p = predict(model, "T0", "T1")
    assert p["margin"] == pytest.approx(p["home_points"] - p["away_points"])
    assert 0 < p["home_win"] < 1
    neutral = predict(model, "T0", "T1", neutral=True)
    assert neutral["margin"] == pytest.approx(p["margin"] - model["home_adv"])
    starter = predict(model, "T0", "T1", home_qb="T0-starter", home_qb_starts=60)
    backup = predict(model, "T0", "T1", home_qb="T0-backup", home_qb_starts=60)
    rookie = predict(model, "T0", "T1", home_qb="Unknown rookie", home_qb_starts=0)
    assert backup["home_win"] < starter["home_win"]
    assert rookie["home_win"] < predict(model, "T0", "T1", home_qb="Unknown veteran", home_qb_starts=100)["home_win"]
    swapped = predict(model, "T1", "T0", neutral=True)
    assert swapped["home_win"] == pytest.approx(1 - neutral["home_win"])


def test_fit_works_without_any_quarterback_information():
    games, _, _ = simulate(seed=2)
    no_qb = games.assign(home_qb=None, away_qb=None, home_qb_starts=np.nan, away_qb_starts=np.nan)
    model = fit(no_qb, "2019-06-01")
    assert model["qb_ratings"].empty and model["inexperience"] == pytest.approx(0, abs=1e-6)


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
