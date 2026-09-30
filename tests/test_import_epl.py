"""The football-data.co.uk importer: parsing quirks, bookmaker mapping, and safe re-runs."""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
import requests

import import_epl
from import_epl import first_value, insert_odds, known_columns, season_code

HEADER = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,B365H,B365D,B365A,PSCH,PSCD,PSCA,B365>2.5,B365<2.5,AHh,B365AHH,B365AHA"


def as_number(text):
    try:
        return float(text)
    except ValueError:
        return text


def test_season_codes():
    assert season_code(2024) == "2425"
    assert season_code(1999) == "9900"


def test_known_columns_keep_similar_bookmaker_codes_apart():
    known = known_columns()
    assert "BFD" in known   # Betfair's draw price (2024/25 files)
    assert "BFDH" in known  # Betfred's home price (2026/27 files)
    assert "PSCH" in known and "B365C>2.5" in known and "AHCh" in known


@pytest.fixture
def db(project):
    conn = sqlite3.connect(project / "data" / "sports.db")
    conn.executescript(Path("schema.sql").read_text())
    yield conn
    conn.close()


def one_game(conn):
    league = import_epl.get_league_id(conn)
    teams = {}
    home = import_epl.get_team_id(conn, league, "Arsenal", teams)
    away = import_epl.get_team_id(conn, league, "Chelsea", teams)
    return import_epl.upsert_game(conn, league, 2024, "2024-08-17", home, away, 2, 1)


def test_rows_longer_than_the_header_are_kept(project, monkeypatch):
    """Regression: 2003/04 and 2004/05 have rows with more fields than the header line."""
    raw = project / "data" / "raw"
    raw.mkdir()
    (raw / "E0_0304.csv").write_text(
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,,\n"
        "E0,16/08/03,Arsenal,Everton,2,1,H,1.5,,\n"
        "E0,16/08/03,Birmingham,Tottenham,1,0,H,2.5,,,1.1,2.2,,\n"
        "E0,17/08/03,Chelsea,Liverpool,1,1,D,2.1\n\n")
    monkeypatch.setattr(import_epl, "RAW_DIR", raw)
    monkeypatch.setattr(import_epl, "LAST_SEASON", 2030)
    df = import_epl.load_season(2003)
    assert list(df["HomeTeam"]) == ["Arsenal", "Birmingham", "Chelsea"]
    assert list(df["Date"]) == ["2003-08-16", "2003-08-16", "2003-08-17"]
    assert any(col.startswith("extra_") for col in df.columns)  # the extra values weren't thrown away


def test_odds_import_maps_markets_and_is_safe_to_rerun(db):
    game_id = one_game(db)
    values = "E0,17/08/2024,Arsenal,Chelsea,2,1,H,10,8,1.9,3.6,4.2,1.95,3.7,4.3,1.8,2.0,-0.5,1.93,1.97".split(",")
    row = pd.Series({col: as_number(v) for col, v in zip(HEADER.split(","), values)})
    first = insert_odds(db, game_id, row)
    second = insert_odds(db, game_id, row)
    assert first == 3 + 3 + 2 + 2   # 1x2 pre-match, 1x2 closing, totals, Asian handicap
    assert second == 0              # nothing duplicated
    markets = dict(db.execute("SELECT market, COUNT(*) FROM odds GROUP BY market").fetchall())
    assert markets == {"1x2": 6, "total": 2, "asian_handicap": 2}
    assert db.execute("SELECT DISTINCT line FROM odds WHERE market = 'asian_handicap'").fetchone()[0] == -0.5


def test_invalid_prices_are_ignored(db):
    game_id = one_game(db)
    row = pd.Series({"B365H": 1.0, "B365D": "n/a", "B365A": None, "PSCH": 2.5})
    assert insert_odds(db, game_id, row) == 1  # only PSCH is a real price
    assert first_value(pd.Series({"AvgH": None, "BbAvH": 2.2}), ["AvgH", "BbAvH"]) == 2.2


def test_full_import_from_cached_files_is_safe_to_rerun(project, monkeypatch):
    raw = project / "data" / "raw"
    raw.mkdir()
    (raw / "E0_2324.csv").write_text(HEADER + "\n"
        "E0,12/08/2023,Arsenal,Chelsea,2,1,H,12,9,1.9,3.6,4.2,1.95,3.7,4.3,1.8,2.0,-0.5,1.93,1.97\n"
        "E0,13/08/2023,Chelsea,Fulham,0,0,D,7,5,1.5,4.0,6.5,1.52,4.1,6.8,1.9,1.9,-1,1.95,1.95\n")
    monkeypatch.setattr(import_epl, "RAW_DIR", raw)
    monkeypatch.setattr(import_epl, "FIRST_SEASON", 2023)
    monkeypatch.setattr(import_epl, "LAST_SEASON", 2024)   # 2024 isn't cached and "downloads" fail
    monkeypatch.setattr(import_epl.requests, "get",
                        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    import_epl.main()
    import_epl.main()
    conn = sqlite3.connect(project / "data" / "sports.db")
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 2
    assert conn.execute("SELECT home_shots FROM soccer_match_stats ORDER BY game_id").fetchall() == [(12,), (7,)]
    assert conn.execute("SELECT COUNT(*) FROM raw_source_rows").fetchone()[0] == 2
