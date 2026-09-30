"""The Understat importer: format checks, team-name aliases, and score cross-checking."""

import sqlite3
from pathlib import Path

import pytest

import import_understat


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        if isinstance(self.payload, Exception):
            raise self.payload
        return self.payload


def match(match_id, home, away, goals, xg, played=True):
    return {"id": str(match_id), "isResult": played, "datetime": "2024-08-16 20:00:00",
            "h": {"title": home}, "a": {"title": away},
            "goals": {"h": str(goals[0]), "a": str(goals[1])}, "xG": {"h": str(xg[0]), "a": str(xg[1])}}


@pytest.fixture
def understat(project, monkeypatch):
    monkeypatch.setattr(import_understat, "CACHE_DIR", project / "data" / "raw" / "understat")
    monkeypatch.setattr(import_understat, "FIRST_SEASON", 2024)
    monkeypatch.setattr(import_understat, "LAST_SEASON", 2024)
    monkeypatch.setattr(import_understat.time, "sleep", lambda s: None)

    def serve(payload):
        monkeypatch.setattr(import_understat.requests, "get", lambda *a, **k: FakeResponse(payload))
    return serve


@pytest.mark.parametrize("payload, message", [
    (ValueError("not json"), "didn't return JSON"),
    ({"teams": {}}, "no 'dates' list"),
    ({"dates": [{"id": "1", "h": {}}]}, "format changed"),
])
def test_format_changes_fail_loudly(understat, payload, message):
    understat(payload)
    with pytest.raises(RuntimeError, match=message):
        import_understat.fetch_season(2024)


def test_links_through_aliases_and_catches_score_mismatches(understat, project, capsys):
    conn = sqlite3.connect(project / "data" / "sports.db")
    conn.executescript(Path("schema.sql").read_text())
    conn.execute("INSERT INTO leagues (code, name, sport) VALUES ('EPL', 'Premier League', 'soccer')")
    for name in ("Man United", "Fulham", "Arsenal", "Wolves"):
        conn.execute("INSERT INTO teams (league_id, name) VALUES (1, ?)", (name,))
    conn.execute("""INSERT INTO games (league_id, season, date, home_team_id, away_team_id, home_score, away_score, status)
                    VALUES (1, 2024, '2024-08-16', 1, 2, 1, 0, 'final'),
                           (1, 2024, '2024-08-17', 3, 4, 2, 2, 'final')""")
    conn.commit()

    understat({"dates": [
        match(101, "Manchester United", "Fulham", (1, 0), (2.4, 0.4)),       # linked via alias
        match(102, "Arsenal", "Wolverhampton Wanderers", (2, 1), (1.8, 0.9)),  # score disagrees: ours 2-2
        match(103, "Arsenal", "Leeds", (0, 0), (1.0, 1.0)),                    # unknown team
        match(104, "Fulham", "Arsenal", (None, None), (None, None), played=False),
    ]})
    import_understat.main()
    out = capsys.readouterr().out

    assert "2 of 3 played matches linked" in out
    assert "Leeds" in out
    assert "Understat 2-1, ours 2-2" in out
    xg = conn.execute("SELECT game_id, home_xg, away_xg FROM game_xg ORDER BY game_id").fetchall()
    assert xg == [(1, 2.4, 0.4), (2, 1.8, 0.9)]
