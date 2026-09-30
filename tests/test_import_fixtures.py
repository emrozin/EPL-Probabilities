"""The fixtures importer: season boundaries, unknown teams, and handing over to results."""

import sqlite3

import pandas as pd

import import_epl
import import_fixtures

FIXTURES = ("\ufeffDiv,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A,BFEH,BFED,BFEA\n"
            "E0,03/10/2026,15:00,Arsenal,Chelsea,1.9,3.6,4.2,1.98,3.75,4.4\n"
            "E0,04/10/2026,14:00,Wrexham,Arsenal,5,4,1.6,,,\n"
            "SP1,04/10/2026,20:00,Barcelona,Getafe,1.3,5,9,,,\n")


class FakeResponse:
    content = FIXTURES.encode("utf-8")

    def raise_for_status(self):
        pass


def test_seasons_start_in_july():
    assert import_fixtures.season_for(pd.Timestamp("2027-03-14")) == 2026
    assert import_fixtures.season_for(pd.Timestamp("2026-08-15")) == 2026


def test_imports_premier_league_fixtures_only_and_hands_over_to_results(project, monkeypatch, capsys):
    conn = sqlite3.connect(project / "data" / "sports.db")
    conn.executescript(open("schema.sql").read())
    league = import_epl.get_league_id(conn)
    for name in ("Arsenal", "Chelsea"):
        conn.execute("INSERT INTO teams (league_id, name) VALUES (?, ?)", (league, name))
    conn.commit()
    monkeypatch.setattr(import_fixtures.requests, "get", lambda *a, **k: FakeResponse())

    import_fixtures.main()
    import_fixtures.main()  # re-running adds nothing
    out = capsys.readouterr().out

    assert "Skipped Wrexham v Arsenal" in out  # unknown team: not silently created
    assert conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0] == 2
    assert conn.execute("SELECT season, date, status, kickoff FROM games").fetchall() == [
        (2026, "2026-10-03", "scheduled", "15:00")]
    assert conn.execute("SELECT COUNT(*) FROM odds WHERE is_closing = 0").fetchone()[0] == 6

    # When the result arrives, the same game row becomes final rather than a duplicate appearing.
    game_id = import_epl.upsert_game(conn, league, 2026, "2026-10-03", 1, 2, 2, 1)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    assert conn.execute("SELECT status, home_score FROM games WHERE id = ?", (game_id,)).fetchone() == ("final", 2)


def test_older_databases_get_the_kickoff_column(project):
    """Databases created before kickoff times were stored are upgraded in place."""
    conn = sqlite3.connect(project / "data" / "sports.db")
    old_schema = open("schema.sql").read().replace(
        "    kickoff       TEXT,                   -- UK local time 'HH:MM', when known (from fixtures.csv)\n", "")
    conn.executescript(old_schema)
    assert "kickoff" not in {row[1] for row in conn.execute("PRAGMA table_info(games)")}
    import_fixtures.ensure_kickoff_column(conn)
    import_fixtures.ensure_kickoff_column(conn)  # safe to run twice
    assert "kickoff" in {row[1] for row in conn.execute("PRAGMA table_info(games)")}
