"""The nflverse importer: relocated teams, playoff rematches, lines, and safe re-runs."""

import sqlite3

import pytest

from nfl import import_nfl
from nfl.import_nfl import american_to_decimal, import_games, read_games

HEADER = ("game_id,season,game_type,week,gameday,gametime,away_team,away_score,home_team,home_score,"
          "location,overtime,home_rest,away_rest,away_moneyline,home_moneyline,spread_line,"
          "away_spread_odds,home_spread_odds,total_line,under_odds,over_odds,div_game,roof,"
          "home_qb_name,away_qb_name,stadium")
ROWS = [
    # 2005: no prices yet, but a spread and total; the Chargers were still in San Diego
    "2005_01_SD_DAL,2005,REG,1,2005-09-11,16:15,SD,24,DAL,28,Home,0,7,7,,,3,,,44,,,0,outdoors,A,B,Texas Stadium",
    # regular season and a playoff rematch: same home team, same opponent, same season
    "2024_10_KC_BUF,2024,REG,10,2024-11-17,16:25,KC,21,BUF,30,Home,0,7,7,110,-130,2.5,-110,-110,46.5,-110,-110,0,outdoors,C,D,Highmark",
    "2024_21_KC_BUF,2024,CON,21,2025-01-26,18:30,KC,32,BUF,29,Home,0,7,7,-105,-115,1,-108,-112,47.5,-110,-110,0,outdoors,C,D,Highmark",
    # a tie, at a neutral site
    "2024_05_NYJ_MIN,2024,REG,5,2024-10-06,09:30,NYJ,20,MIN,20,Neutral,1,7,7,140,-165,2.5,-110,-110,41,-110,-110,0,outdoors,E,F,Tottenham",
    # not played yet
    "2026_07_SF_LA,2026,REG,7,2026-10-18,16:05,SF,,LA,,Home,,7,7,-120,100,-1.5,-110,-110,48.5,-110,-110,1,dome,G,H,SoFi",
]


def csv_text(rows=ROWS):
    return "\n".join([HEADER, *rows]) + "\n"


@pytest.fixture
def conn(project):
    connection = sqlite3.connect(project / "data" / "sports.db")
    yield connection
    connection.close()


def test_american_odds_convert_to_decimal():
    assert american_to_decimal(-150) == pytest.approx(1 + 100 / 150)
    assert american_to_decimal(130) == pytest.approx(2.30)
    assert american_to_decimal(100) == pytest.approx(2.0)
    assert american_to_decimal(None) is None and american_to_decimal(float("nan")) is None


def test_imports_games_with_relocated_teams_as_one_franchise(conn):
    counts = import_games(conn, read_games(csv_text()))
    assert counts["new"] == 5 and counts["final"] == 4 and counts["scheduled"] == 1
    teams = {name for (name,) in conn.execute("SELECT name FROM teams")}
    assert "LAC" in teams and "SD" not in teams


def test_playoff_rematch_is_a_separate_game(conn):
    import_games(conn, read_games(csv_text()))
    games = conn.execute("""SELECT i.game_type, g.home_score, g.away_score FROM games g
                            JOIN nfl_game_info i ON i.game_id = g.id
                            WHERE i.source_game_id LIKE '2024_%_KC_BUF' ORDER BY g.date""").fetchall()
    assert games == [("REG", 30, 21), ("CON", 29, 32)]


def test_lines_are_stored_from_the_home_teams_side(conn):
    import_games(conn, read_games(csv_text()))
    game = conn.execute("SELECT game_id FROM nfl_game_info WHERE source_game_id = '2024_10_KC_BUF'").fetchone()[0]
    odds = {(m, o): (round(price, 3), line) for m, o, price, line in conn.execute(
        "SELECT market, outcome, price, line FROM odds WHERE game_id = ?", (game,))}
    assert odds[("spread", "home")][1] == -2.5          # BUF favored by 2.5: a -2.5 handicap for the home team
    assert odds[("moneyline", "home")] == (round(1 + 100 / 130, 3), None)
    assert odds[("total", "over")][1] == 46.5
    spread, = conn.execute("SELECT spread_line FROM nfl_game_info WHERE game_id = ?", (game,)).fetchone()
    assert spread == 2.5                                 # nflverse's own convention, kept as is


def test_spread_is_kept_even_before_prices_exist(conn):
    import_games(conn, read_games(csv_text()))
    game, spread, total = conn.execute("SELECT game_id, spread_line, total_line FROM nfl_game_info "
                                       "WHERE source_game_id = '2005_01_SD_DAL'").fetchone()
    assert (spread, total) == (3, 44)
    assert conn.execute("SELECT COUNT(*) FROM odds WHERE game_id = ?", (game,)).fetchone()[0] == 0


def test_ties_neutral_sites_and_scheduled_games(conn):
    import_games(conn, read_games(csv_text()))
    tie = conn.execute("""SELECT g.status, g.home_score, g.away_score, i.neutral_site, i.overtime FROM games g
                          JOIN nfl_game_info i ON i.game_id = g.id WHERE i.source_game_id = '2024_05_NYJ_MIN'""").fetchone()
    assert tie == ("final", 20, 20, 1, 1)
    upcoming = conn.execute("""SELECT g.status, g.home_score, g.kickoff FROM games g JOIN nfl_game_info i
                               ON i.game_id = g.id WHERE i.source_game_id = '2026_07_SF_LA'""").fetchone()
    assert upcoming == ("scheduled", None, "16:05")
    assert conn.execute("SELECT COUNT(*) FROM odds o JOIN nfl_game_info i ON i.game_id = o.game_id "
                        "WHERE i.source_game_id = '2026_07_SF_LA'").fetchone()[0] == 0


def test_rerun_updates_games_without_duplicates(conn):
    import_games(conn, read_games(csv_text()))
    played = [r.replace("2026-10-18,16:05,SF,,LA,,", "2026-10-19,20:15,SF,17,LA,24,") for r in ROWS]  # moved and played
    counts = import_games(conn, read_games(csv_text(played)))
    assert counts["new"] == 0 and counts["updated"] == 5
    assert conn.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 5
    game = conn.execute("""SELECT g.date, g.kickoff, g.status, g.home_score FROM games g JOIN nfl_game_info i
                           ON i.game_id = g.id WHERE i.source_game_id = '2026_07_SF_LA'""").fetchone()
    assert game == ("2026-10-19", "20:15", "final", 24)
    assert conn.execute("SELECT COUNT(*) FROM odds o JOIN nfl_game_info i ON i.game_id = o.game_id "
                        "WHERE i.source_game_id = '2026_07_SF_LA'").fetchone()[0] == 6   # lines once played


def test_a_changed_file_format_fails_loudly():
    with pytest.raises(SystemExit, match="spread_line"):
        read_games(csv_text().replace("spread_line", "spread"))


def test_nfl_and_soccer_share_the_database(league_db):
    conn = sqlite3.connect(league_db)
    import_games(conn, read_games(csv_text()))
    leagues = dict(conn.execute("SELECT code, sport FROM leagues"))
    assert leagues == {"EPL": "soccer", "NFL": "american_football"}
    nfl_teams = conn.execute("SELECT COUNT(*) FROM teams WHERE league_id = (SELECT id FROM leagues "
                             "WHERE code = 'NFL')").fetchone()[0]
    assert nfl_teams == 8   # LAC, DAL, KC, BUF, NYJ, MIN, SF, LA
    conn.close()


def test_main_downloads_and_imports(project, monkeypatch, capsys):
    class Response:
        content = csv_text().encode()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(import_nfl.requests, "get", lambda *a, **k: Response())
    import_nfl.main()
    assert "5 NFL games in the file" in capsys.readouterr().out
    assert (project / "data" / "raw" / "nfl_games.csv").exists()
