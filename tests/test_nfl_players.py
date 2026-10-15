"""NFL stats import and player projections: sensible numbers, the listed quarterback, no look-ahead."""

import sqlite3

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import nfl.site as nfl_site
from nfl import import_nfl_stats, predict_players, predict_upcoming
from nfl.players import Projector, Settings, evaluate, load, load_injuries, total_error
from test_nfl_site import NOW, nfl_project  # noqa: F401  (the simulated NFL league fixture)


def add_stats(db_path, seed=0, only=None):
    """Simulated team and player stats for every finished game: each team has a starting QB, two
    receivers (60% / 40% of targets) and a running back."""
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(db_path)
    games = pd.read_sql("""SELECT g.id, h.name AS home, a.name AS away, i.home_qb, i.away_qb FROM games g
                           JOIN teams h ON h.id = g.home_team_id JOIN teams a ON a.id = g.away_team_id
                           JOIN nfl_game_info i ON i.game_id = g.id WHERE g.status = 'final'""", conn)
    if only is not None:
        games = games[games["id"].isin(only)]
    team_rows, player_rows = [], []
    for g in games.itertuples():
        for team, qb in ((g.home, g.home_qb), (g.away, g.away_qb)):
            att, car = int(rng.integers(28, 40)), int(rng.integers(20, 30))
            wr1, wr2 = round(att * 0.6), att - round(att * 0.6)
            yds = int(att * 7)
            team_rows.append((g.id, team, att, 2, car, yds, car * 4, 1.0, 0.5))
            player_rows += [
                (g.id, f"{qb}-id", qb, "QB", team, 20, att, yds, 1, 1, 2, 5, 0, 0, 0, 0, 0, 0),
                (g.id, f"{team}-wr1", f"{team} Receiver", "WR", team, 0, 0, 0, 0, 0, 0, 0, 0, wr1, int(wr1 * .65), wr1 * 8, 1, wr1 * 11),
                (g.id, f"{team}-wr2", f"{team} Slot", "WR", team, 0, 0, 0, 0, 0, 0, 0, 0, wr2, int(wr2 * .65), wr2 * 7, 0, wr2 * 6),
                (g.id, f"{team}-rb", f"{team} Back", "RB", team, 0, 0, 0, 0, 0, car - 2, (car - 2) * 4, 0, 0, 0, 0, 0, 0)]
    conn.executemany("INSERT INTO nfl_team_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", team_rows)
    conn.executemany("INSERT INTO nfl_player_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", player_rows)
    conn.commit()
    conn.close()


@pytest.fixture
def stats_project(nfl_project):
    add_stats(nfl_project / "data" / "sports.db")
    return nfl_project


def test_stats_import_maps_games_and_relocated_teams_and_keeps_offense_only(project):
    conn = sqlite3.connect(project / "data" / "sports.db")
    conn.executescript(open("schema.sql").read())
    player = pd.DataFrame([
        {"game_id": "2019_01_OAK_DEN", "team": "OAK", "player_id": "p1", "player_display_name": None, "position": "QB",
         **{c: 1 for c in import_nfl_stats.PLAYER_COLUMNS}},
        {"game_id": "2019_01_OAK_DEN", "team": "DEN", "player_id": "p2", "player_display_name": "Line Backer",
         "position": "LB", **{c: 0 for c in import_nfl_stats.PLAYER_COLUMNS}},
        {"game_id": "not_a_game", "team": "DEN", "player_id": "p3", "player_display_name": "X", "position": "WR",
         **{c: 1 for c in import_nfl_stats.PLAYER_COLUMNS}},
    ])
    stored = import_nfl_stats.store(conn, "player", player, {"2019_01_OAK_DEN": 7})
    rows = conn.execute("SELECT game_id, team, name FROM nfl_player_stats").fetchall()
    conn.close()
    assert stored == 1 and rows == [(7, "LV", "p1")]   # Raiders -> LV; linebacker and unknown game left out


def test_current_season_starts_in_july():
    from datetime import date
    assert import_nfl_stats.current_season(date(2026, 10, 14)) == 2026
    assert import_nfl_stats.current_season(date(2027, 1, 20)) == 2026


def test_projections_follow_target_shares_and_use_the_listed_quarterback(stats_project):
    conn = sqlite3.connect(stats_project / "data" / "sports.db")
    games, teams, players = load(conn)
    conn.close()
    week6 = games[(games["season"] == 2026) & (games["week"] == 6)]
    proj = Projector(games, teams, players).run({gid: (24.0, 20.0) for gid in week6["id"]})
    game = week6.iloc[0]
    home = proj[(proj["game_id"] == game["id"]) & (proj["team"] == game["home"])].set_index("position")
    wr = proj[(proj["game_id"] == game["id"]) & (proj["team"] == game["home"]) & (proj["position"] == "WR")]
    receiver = wr.loc[wr["name"].str.endswith("Receiver"), "targets"].iloc[0]
    slot = wr.loc[wr["name"].str.endswith("Slot"), "targets"].iloc[0]
    assert receiver / (receiver + slot) == pytest.approx(0.6, abs=0.03)
    assert home.loc["QB", "name"] == game["home_qb"]        # the listed starter, even a backup
    assert 150 < home.loc["QB", "pass_yds"] < 350
    assert ((proj["anytime_td"] >= 0) & (proj["anytime_td"] < 1)).all()


def test_projections_never_use_the_game_being_projected(stats_project):
    conn = sqlite3.connect(stats_project / "data" / "sports.db")
    games, teams, players = load(conn)
    conn.close()
    target = games[(games["season"] == 2025) & (games["week"] == 10)]["id"].iloc[0]
    before = Projector(games, teams, players).run({target: (24.0, 20.0)})
    changed = players.copy()
    changed.loc[changed["game_id"] == target, ["receiving_yards", "passing_yards"]] = 999
    after = Projector(games, teams, changed).run({target: (24.0, 20.0)})
    pd.testing.assert_frame_equal(before, after)


def test_evaluate_compares_with_benchmarks(stats_project):
    conn = sqlite3.connect(stats_project / "data" / "sports.db")
    games, teams, players = load(conn)
    conn.close()
    targets = {gid: (24.0, 20.0) for gid in games[(games["season"] == 2025) & (games["status"] == "final")]["id"]}
    result = evaluate(Projector(games, teams, players, Settings()).run(targets), players, games).set_index("stat")
    assert set(result.index) >= {"pass_yds", "rec_yds", "receptions", "rush_yds"}
    assert (result["players"] > 0).all()


def test_live_projections_are_recorded_and_shown_on_the_game_page(stats_project, monkeypatch):
    predict_players.main(["--record"], now=NOW)
    record = pd.read_csv(predict_players.RECORD_PATH)
    assert len(record) > 0 and set(record["game"].str[:7]) == {"2026_06", "2026_07"}
    assert (record.groupby(["game", "team"])["position"].apply(lambda s: (s == "QB").sum()) == 1).all()

    import app
    nfl_site._cache.clear()
    game = record["game"].iloc[0].lower().replace("_", "-")
    page = TestClient(app.app).get(f"/nfl/game/{game}/").text
    assert "Player projections" in page and "Pass yds" in page


def test_projections_for_started_games_never_change(stats_project):
    predict_players.main(["--record"], now=NOW)
    before = pd.read_csv(predict_players.RECORD_PATH)
    from datetime import datetime
    later = datetime(2026, 10, 15, 14, 0, tzinfo=predict_upcoming.EASTERN)
    predict_players.main(["--record"], now=later)
    after = pd.read_csv(predict_players.RECORD_PATH)
    started = before[before["date"] == "2026-10-15"]
    assert len(started) > 0
    pd.testing.assert_frame_equal(started.reset_index(drop=True),
                                  after[after["date"] == "2026-10-15"].reset_index(drop=True))


def test_track_record_compares_live_player_projections_with_what_happened(stats_project):
    predict_players.main(["--record"], now=NOW)
    db = stats_project / "data" / "sports.db"
    conn = sqlite3.connect(db)
    week6 = [r[0] for r in conn.execute("""SELECT g.id FROM games g JOIN nfl_game_info i ON i.game_id = g.id
                                           WHERE g.season = 2026 AND i.week = 6""")]
    conn.executemany("UPDATE games SET status = 'final', home_score = 24, away_score = 17 WHERE id = ?",
                     [(g,) for g in week6])
    conn.commit()
    conn.close()
    add_stats(db, seed=1, only=week6)

    import app
    nfl_site._cache.clear()
    page = TestClient(app.app).get("/nfl/track-record/").text
    assert "Player projections" in page and "Receiving yards" in page and "Biggest surprises" in page
    track = nfl_site.player_track()
    assert track["games"] == len(week6)
    assert {r["label"] for r in track["overall"]} >= {"Passing yards", "Receiving yards", "Rushing yards"}


def test_injury_reports_remove_ruled_out_players_and_bring_back_returners(stats_project):
    db = stats_project / "data" / "sports.db"
    conn = sqlite3.connect(db)
    games, teams, players = load(conn)
    week6 = games[(games["season"] == 2026) & (games["week"] == 6)].iloc[0]
    team = week6["home"]
    week5 = games[(games["season"] == 2026) & (games["week"] == 5) & ((games["home"] == team) | (games["away"] == team))]
    # The slot receiver missed week 5 while on the report; this week he's Questionable. The No. 1 receiver is Out.
    players = players[~((players["game_id"].isin(week5["id"])) & (players["player_id"] == f"{team}-wr2"))]
    injuries = {(2026, 5, team): {f"{team}-wr2": "Out"},
                (2026, 6, team): {f"{team}-wr1": "Out", f"{team}-wr2": "Questionable"}}
    conn.close()
    targets = {week6["id"]: (24.0, 20.0)}
    with_reports = Projector(games, teams, players, injuries=injuries).run(targets)
    without = Projector(games, teams, players).run(targets)
    names = lambda df: set(df.loc[df["team"] == team, "name"])
    assert f"{team} Receiver" in names(without) and f"{team} Slot" not in names(without)
    assert f"{team} Receiver" not in names(with_reports)      # ruled out
    assert f"{team} Slot" in names(with_reports)              # back from injury


def test_total_error_counts_unprojected_players_and_players_who_did_not_play():
    projections = pd.DataFrame([{"game_id": 1, "player_id": "a", "pass_yds": 0, "rush_yds": 10, "receptions": 2, "rec_yds": 20},
                                {"game_id": 1, "player_id": "b", "pass_yds": 0, "rush_yds": 0, "receptions": 3, "rec_yds": 30}])
    players = pd.DataFrame([{"game_id": 1, "player_id": "a", "position": "WR", "passing_yards": 0, "rushing_yards": 10,
                             "receptions": 2, "receiving_yards": 25},
                            {"game_id": 1, "player_id": "c", "position": "RB", "passing_yards": 0, "rushing_yards": 40,
                             "receptions": 1, "receiving_yards": 5}])
    err = total_error(projections, players, [1])
    # a: off by 5; b projected 30 but didn't play; c played but wasn't projected (5). Divided by 2 players who played.
    assert err["rec_yds"] == pytest.approx((5 + 30 + 5) / 2)


def test_ranges_from_the_repository_are_sensible():
    from nfl.players import load_ranges, stat_range
    table = load_ranges()
    assert table is not None, "nfl/ranges.json is missing"
    for stat, projected in (("rec_yds", 50), ("rush_yds", 60), ("pass_yds", 230), ("receptions", 4)):
        low, high = stat_range(table, stat, projected)
        assert 0 <= low < projected < high
    assert stat_range(table, "rec_yds", 80)[1] > stat_range(table, "rec_yds", 20)[1]


def test_fit_ranges_covers_about_80_percent_of_its_own_data():
    from nfl.players import coverage, fit_ranges
    rng = np.random.default_rng(3)
    proj = rng.uniform(5, 90, 5000)
    df = pd.DataFrame({"proj_rec_yds": proj, "receiving_yards": proj * rng.gamma(2.0, 0.5, 5000),
                       "proj_rush_yds": proj, "rushing_yards": proj * rng.gamma(3.0, 1 / 3, 5000),
                       "proj_pass_yds": proj * 3, "passing_yards": proj * 3 * rng.gamma(8.0, 1 / 8, 5000),
                       "proj_receptions": proj / 15, "receptions": proj / 15 * rng.gamma(4.0, 0.25, 5000)})
    cov = coverage(fit_ranges(df), df)
    assert all(0.75 < v < 0.85 for v in cov.values())
