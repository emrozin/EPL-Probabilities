"""Playoff odds: the season simulator, its tiebreakers and the playoff odds page."""

import sqlite3

import numpy as np
import pandas as pd
import pytest

import nfl.site as nfl_site
from nfl import season
from nfl.model import add_qb_changes
from nfl.teams import TEAMS, conference
from test_nfl_site import nfl_project, client  # noqa: F401  (the simulated NFL league fixtures)


def full_league(seed=0, seasons=(2024, 2025, 2026), played_weeks=None) -> pd.DataFrame:
    """All 32 teams; each plays the other 15 teams in its conference once a season. In the last
    season, only the first `played_weeks` weeks have been played."""
    rng = np.random.default_rng(seed)
    teams = sorted(TEAMS)
    strength = dict(zip(teams, rng.normal(0, 5, len(teams))))
    rows, gid = [], 0
    for s in seasons:
        pairs = [(a, b) for i, a in enumerate(teams) for b in teams[i + 1:] if conference(a) == conference(b)]
        rng.shuffle(pairs)
        for k, (a, b) in enumerate(pairs):
            week = k // 16 + 1
            home, away = (a, b) if rng.random() < 0.5 else (b, a)
            played = played_weeks is None or s < max(seasons) or week <= played_weeks
            margin = 2 + strength[home] - strength[away] + rng.normal(0, 13)
            gid += 1
            rows.append({"id": gid, "season": s, "date": str(pd.Timestamp(f"{s}-09-07") + pd.Timedelta(days=7 * (week - 1)))[:10],
                         "status": "final" if played else "scheduled", "home": home, "away": away,
                         "home_score": max(0, round(21 + margin / 2)) if played else np.nan,
                         "away_score": max(0, round(21 - margin / 2)) if played else np.nan,
                         "neutral_site": 0, "game_type": "REG", "week": week, "spread_line": np.nan,
                         "home_qb": f"{home} QB" if played else None, "away_qb": f"{away} QB" if played else None})
    games = pd.DataFrame(rows)
    games.loc[games["home_score"] == games["away_score"], "home_score"] += 1   # no ties
    return add_qb_changes(games)


def test_every_simulation_has_the_right_number_of_winners():
    games = full_league(played_weeks=6)
    s = season.setup(games, 2026, games.loc[(games["season"] == 2026) & (games["week"] == 7), "date"].min())
    odds = season.simulate(s, n=300)
    assert odds["division"].sum() == pytest.approx(8)
    assert odds["playoffs"].sum() == pytest.approx(14)
    assert odds["top_seed"].sum() == pytest.approx(2)
    assert odds["conference"].sum() == pytest.approx(2)
    assert odds["champion"].sum() == pytest.approx(1)
    assert (odds["division"] <= odds["playoffs"] + 1e-9).all()
    assert 7 < odds["wins"].mean() < 8   # 15 games each, no ties


def test_a_finished_season_is_settled_by_the_standings_and_tiebreakers():
    games = full_league()
    s = season.setup(games, 2026, "2027-02-01")      # every regular-season game known
    odds = season.simulate(s, n=20)
    assert set(odds["playoffs"].round(6)) <= {0.0, 1.0} and set(odds["division"].round(6)) <= {0.0, 1.0}
    reg = games[games["season"] == 2026]
    wins = pd.concat([reg.loc[reg["home_score"] > reg["away_score"], "home"],
                      reg.loc[reg["away_score"] > reg["home_score"], "away"]]).value_counts()
    for div in sorted({d for _, d in TEAMS.values()}):
        members = [t for t, (_, d) in TEAMS.items() if d == div]
        best = max(wins.get(t, 0) for t in members)
        leaders = [t for t in members if wins.get(t, 0) == best]
        winner = odds.set_index("team").loc[members, "division"].idxmax()
        assert winner in leaders
        if len(leaders) == 2:   # a two-way tie goes to the head-to-head winner
            a, b = leaders
            game = reg[((reg["home"] == a) & (reg["away"] == b)) | ((reg["home"] == b) & (reg["away"] == a))].iloc[0]
            h2h_winner = game["home"] if game["home_score"] > game["away_score"] else game["away"]
            assert winner == h2h_winner


def test_six_team_format_before_2020():
    games = full_league(seasons=(2016, 2017, 2018), played_weeks=8)
    s = season.setup(games, 2018, games.loc[(games["season"] == 2018) & (games["week"] == 9), "date"].min())
    odds = season.simulate(s, n=200)
    assert odds["playoffs"].sum() == pytest.approx(12)


def test_actual_outcomes_read_from_the_playoff_games():
    games = pd.DataFrame([
        {"season": 2025, "game_type": "REG", "home": "KC", "away": "BUF", "home_score": 20, "away_score": 10, "status": "final"},
        {"season": 2025, "game_type": "WC", "home": "BUF", "away": "MIA", "home_score": 20, "away_score": 10, "status": "final"},
        {"season": 2025, "game_type": "DIV", "home": "KC", "away": "BUF", "home_score": 30, "away_score": 10, "status": "final"},
        {"season": 2025, "game_type": "SB", "home": "KC", "away": "SEA", "home_score": 30, "away_score": 31, "status": "final"},
    ])
    actual = season.actual_outcomes(games, 2025).set_index("team")
    assert actual.at["KC", "division"] and actual.at["BUF", "division"]    # a bye, and a wild card host
    assert actual.at["KC", "conference"] and not actual.at["BUF", "conference"]


def test_playoff_odds_page(client, nfl_project):
    conn = sqlite3.connect(nfl_project / "data" / "sports.db")
    conn.executescript(open("schema.sql").read())
    teams = ["KC", "BUF", "SEA", "DEN", "SF", "DAL", "GB", "MIA"]
    for as_of, shift in (("2026-10-07", 0.0), ("2026-10-14", 0.1)):
        conn.executemany("INSERT INTO nfl_season_odds VALUES (2026, ?, ?, 9.5, 0.3, ?, 0.1, 0.05, 0.02)",
                         [(as_of, t, min(1.0, 0.4 + shift)) for t in teams])
    conn.commit()
    conn.close()
    nfl_site._cache.clear()
    page = client.get("/nfl/playoffs/").text
    assert "Playoff odds: 2026" in page and "Kansas City Chiefs" in page and "+10" in page
    assert "Playoff chance through 2026" in client.get("/nfl/team/kansas-city-chiefs/").text
    assert nfl_site.odds_text(0) == "–" and nfl_site.odds_text(0.003) == "<1%" and nfl_site.odds_text(0.999) == ">99%"


def test_playoff_odds_page_without_odds(client):
    assert "No playoff odds yet" in client.get("/nfl/playoffs/").text
