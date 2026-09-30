"""The website: every page loads, missing pages 404, and every link leads somewhere real."""

import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app as site
import run_backtest


@pytest.fixture
def client(league_db, monkeypatch):
    monkeypatch.setattr(site, "DB_PATH", league_db)
    site._cache.clear()
    run_backtest.main()  # writes backtest predictions into the test database
    return TestClient(site.app)


def test_every_page_loads(client):
    for path in ["/", "/results/", "/table/", "/table/2020/", "/team/team-a/", "/team/team-a/all/",
                 "/team/team-a/2019/", "/backtest/", "/ratings/"]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert "Premier League probabilities" in response.text


def test_missing_pages_show_the_not_found_page(client):
    for path in ["/team/nobody/", "/team/team-a/1900/", "/table/1900/", "/results/1900-01-01/", "/no-such-page/"]:
        response = client.get(path)
        assert response.status_code == 404, path
        assert "Page not found" in response.text


def test_every_link_on_the_site_works(client):
    """Follow every internal link from the home page, exactly as the static site build does."""
    queue, seen = ["/"], {"/"}
    while queue:
        path = queue.pop()
        response = client.get(path)
        assert response.status_code == 200, f"broken link to {path}"
        for link in re.findall(r'(?:href|value)="(/[^"#?]*)"', response.text):
            if not link.startswith("/static/") and link not in seen:
                seen.add(link)
                queue.append(link)
    assert len(seen) > 50  # it really did crawl the site


def test_results_page_highlights_the_winner(client):
    html = client.get("/results/").text
    assert 'class="teams finished"' in html
    assert "winner" in html or "level" in html


def test_percentages_always_add_up_to_100():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        p = rng.dirichlet([1, 1, 1])
        shown = site.percentages(*p)
        assert sum(shown.values()) == 100
        assert all(abs(shown[o] - q * 100) < 1 for o, q in zip(("home", "draw", "away"), p))


def test_team_slugs():
    assert site.slugify("Nott'm Forest") == "nottm-forest"
    assert site.slugify("Man United") == "man-united"
    assert site.slugify("Brighton & Hove Albion") == "brighton-and-hove-albion"


def test_league_table_ranks_by_points_then_goal_difference_then_goals():
    games = pd.DataFrame([
        # A: win + loss (3 pts, GD 0, GF 4).  B: loss + win (3 pts, GD +1, GF 3).  C: 2 losses... etc.
        {"season": 2024, "date": "2024-08-10", "home": "A", "away": "B", "home_score": 3, "away_score": 1},
        {"season": 2024, "date": "2024-08-17", "home": "B", "away": "A", "home_score": 2, "away_score": 1},
        {"season": 2024, "date": "2024-08-24", "home": "B", "away": "C", "home_score": 0, "away_score": 0},
        {"season": 2024, "date": "2024-08-24", "home": "C", "away": "A", "home_score": 0, "away_score": 0},
    ]).assign(home_xg=np.nan, away_xg=np.nan)
    table = site.league_table(games, 2024).set_index("team")
    assert table.loc["A", "points"] == 4 and table.loc["B", "points"] == 4 and table.loc["C", "points"] == 2
    assert table.loc["A", "gd"] == 1 and table.loc["B", "gd"] == -1
    assert list(table.sort_values("position").index) == ["A", "B", "C"]  # level on points: A's better GD wins
    assert table.loc["A", "form"] == ["W", "L", "D"]
