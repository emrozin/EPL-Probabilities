"""College football: the importer (including matching betting lines to teams), the model, the
prediction record and the website pages."""

import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import cfb.site as cfb_site
from cfb import import_cfb, predict_upcoming, records
from cfb.model import backtest, fit, load_games, predict
from cfb.run_backtest import save

COLUMNS = ["game_id", "season", "week", "season_type", "start_date", "start_time_tbd", "completed", "neutral_site",
           "conference_game", "home_id", "home_team", "home_division", "home_conference", "home_points", "home_pregame_elo",
           "away_id", "away_team", "away_division", "away_conference", "away_points", "away_pregame_elo", "notes", "venue"]
TEAMS = [("Michigan", "Big Ten"), ("Ohio State", "Big Ten"), ("Alabama", "SEC"), ("Georgia", "SEC"),
         ("Texas", "SEC"), ("Oregon", "Big Ten"), ("Clemson", "ACC"), ("Miami", "ACC")]
FCS = [("Furman", "Southern"), ("Montana", "Big Sky")]
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=predict_upcoming.EASTERN)


def simulated_schedule(seed=0) -> pd.DataFrame:
    """Four seasons of a small league plus two FCS schools; 2026 is played through week 6."""
    rng = np.random.default_rng(seed)
    strength = {t: s for (t, _), s in zip(TEAMS + FCS, rng.normal(0, 10, len(TEAMS) + len(FCS)))}
    for t, _ in FCS:
        strength[t] -= 20
    ids = {t: 100 + i for i, (t, _) in enumerate(TEAMS + FCS)}
    conf = dict(TEAMS + FCS)
    rows, gid = [], 400000000
    for season in range(2023, 2027):
        for week in range(1, 13):
            order = list(rng.permutation([t for t, _ in TEAMS]))
            pairs = [(order[2 * k], order[2 * k + 1]) for k in range(4)]
            if week == 1:   # FBS teams host FCS schools in week 1
                pairs = [(order[0], "Furman"), (order[1], "Montana")] + pairs[1:3]
            for home, away in pairs:
                gid += 1
                played = season < 2026 or week <= 6
                margin = 3 + strength[home] - strength[away] + rng.normal(0, 14)
                day = pd.Timestamp(f"{season}-08-29") + pd.Timedelta(days=7 * (week - 1))
                rows.append({"game_id": gid, "season": season, "week": week, "season_type": "regular",
                             "start_date": f"{day:%Y-%m-%d}T23:30:00.000Z", "start_time_tbd": "FALSE",
                             "completed": "TRUE" if played else "FALSE", "neutral_site": "FALSE", "conference_game": "FALSE",
                             "home_id": ids[home], "home_team": home, "home_division": "fbs" if home not in dict(FCS) else "fcs",
                             "home_conference": conf[home], "home_points": max(0, round(28 + margin / 2)) if played else np.nan,
                             "home_pregame_elo": 1500 + 25 * strength[home], "away_id": ids[away], "away_team": away,
                             "away_division": "fcs" if away in dict(FCS) else "fbs", "away_conference": conf[away],
                             "away_points": max(0, round(28 - margin / 2)) if played else np.nan,
                             "away_pregame_elo": 1500 + 25 * strength[away], "notes": np.nan, "venue": "Stadium"})
    df = pd.DataFrame(rows, columns=COLUMNS)
    tie = df["home_points"] == df["away_points"]
    df.loc[tie, "home_points"] += 1
    return df


def simulated_lines(schedule: pd.DataFrame) -> pd.DataFrame:
    """Two books' moneylines for the finished games; team names abbreviated, as in early seasons."""
    abbr = {t: t[:3].upper() for t, _ in TEAMS + FCS}
    rows = []
    for g in schedule[schedule["completed"] == "TRUE"].itertuples():
        for book, shade in (("A", 0.0), ("B", 0.02)):
            p = min(0.95, max(0.05, 0.6 + shade))
            to_american = lambda q: -100 * q / (1 - q) if q > 0.5 else 100 * (1 - q) / q
            for team, q in ((g.home_team, p * 1.02), (g.away_team, (1 - p) * 1.02)):
                rows.append({"game_id": g.game_id, "season": g.season, "market_type": "money_line", "abbr": abbr[team],
                             "lines": np.nan, "odds": to_american(q), "book": book})
            rows.append({"game_id": g.game_id, "season": g.season, "market_type": "spread", "abbr": abbr[g.home_team],
                         "lines": -3.5, "odds": np.nan, "book": book})
    return pd.DataFrame(rows)


@pytest.fixture
def cfb_project(project, monkeypatch):
    db = project / "data" / "sports.db"
    schedule = simulated_schedule()
    lines = import_cfb.consensus_lines(simulated_lines(schedule), schedule)
    conn = sqlite3.connect(db)
    import_cfb.import_games(conn, schedule, lines)
    games = load_games(conn)
    save(conn, backtest(games, 2024), "cfb_points_backtest", "v1")
    conn.close()
    predict_upcoming.main(["--record"], now=NOW)
    monkeypatch.setattr(cfb_site, "DB_PATH", db)
    monkeypatch.setattr(cfb_site, "date", type("FrozenDate", (), {"today": staticmethod(lambda: datetime(2026, 10, 14).date())}))
    cfb_site._cache.clear()
    return project


def test_import_keeps_fbs_and_fcs_games_and_matches_lines_to_the_right_team(cfb_project):
    conn = sqlite3.connect(cfb_project / "data" / "sports.db")
    total, fcs = conn.execute("""SELECT COUNT(*), SUM(home_division = 'fcs' OR away_division = 'fcs') FROM cfb_game_info""").fetchone()
    assert total == 4 * 12 * 4 and fcs == 8
    market = pd.read_sql("SELECT 1 / price AS p FROM odds WHERE bookmaker = 'consensus' AND outcome = 'home'", conn)["p"]
    assert market.round(3).eq(0.61).all()          # the average of 60% and 62%, with the margins removed
    spread = conn.execute("SELECT DISTINCT spread_line FROM cfb_game_info WHERE spread_line IS NOT NULL").fetchall()
    assert spread == [(-3.5,)]
    conn.close()


def test_kickoffs_are_in_eastern_time():
    assert import_cfb.kickoff_eastern("2026-10-17T23:30:00.000Z") == ("2026-10-17", "19:30")
    assert import_cfb.kickoff_eastern("2026-10-18T01:00:00.000Z") == ("2026-10-17", "21:00")


def test_model_rates_strong_teams_higher_and_fcs_schools_low(cfb_project):
    conn = sqlite3.connect(cfb_project / "data" / "sports.db")
    games = load_games(conn)
    conn.close()
    played = games[games["status"] == "final"]
    model = fit(played, "2026-10-14", 2026)
    overall = model["ratings"]["offense"] + model["ratings"]["defense"]
    assert overall[["Furman", "Montana"]].max() < overall.drop(["Furman", "Montana"]).min()
    best = overall.idxmax()
    assert predict(model, best, "Furman")["home_win"] > 0.75 > predict(model, "Furman", best)["home_win"]
    assert abs(model["home_adv"]) < 15   # a tiny league can't pin it down; real data gives about 3 points


def test_live_predictions_are_recorded_and_not_changed_after_kickoff(cfb_project):
    record = records.read_record()
    assert len(record) == 4 and np.allclose(record["home_win"] + record["away_win"], 1)   # week 7's four games
    later = datetime(2026, 10, 17, 20, 0, tzinfo=predict_upcoming.EASTERN)                # after the 19:30 kickoffs
    predict_upcoming.main(["--record"], now=later)
    after = records.read_record()
    assert len(after) == 8                                                                 # week 8 added
    pd.testing.assert_frame_equal(after[after["game"].isin(record["game"])].reset_index(drop=True), record)


@pytest.mark.parametrize("path", ["/cfb/", "/cfb/?conference=SEC", "/cfb/results/", "/cfb/results/2025/3/",
                                  "/cfb/ratings/", "/cfb/ratings/?conference=ACC", "/cfb/team/michigan/",
                                  "/cfb/team/michigan/2025/", "/cfb/track-record/", "/cfb/backtest/"])
def test_cfb_pages_load(cfb_project, path):
    import app
    response = TestClient(app.app).get(path)
    assert response.status_code == 200
    assert "College football probabilities" in response.text


def test_upcoming_page_and_conference_filter(cfb_project):
    import app
    client = TestClient(app.app)
    page = client.get("/cfb/").text
    assert page.count('class="match card') == 4
    sec = client.get("/cfb/?conference=SEC").text
    assert 0 < sec.count('class="match card') < 4 or "No upcoming games predicted for the SEC" in sec
    assert client.get("/cfb/team/not-a-team/").status_code == 404
