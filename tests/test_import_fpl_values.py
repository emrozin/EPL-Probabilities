"""Squad values from FPL prices: teams from the matches themselves, and only prices known on each date."""

import pandas as pd
import pytest

from import_fpl_values import gameweek_values, player_rows, team_mapping, weekly_values


def gw_row(gw, fixture, element, value, was_home, opponent, kickoff, h_score=1, a_score=0):
    return {"GW": gw, "fixture": fixture, "element": element, "value": value, "was_home": was_home,
            "opponent_team": opponent, "kickoff_time": kickoff, "team_h_score": h_score, "team_a_score": a_score}


@pytest.fixture
def rows():
    # Gameweek 1: team 1 at home to team 2. Gameweek 2: team 2 at home to team 1.
    data = [gw_row(1, 10, e, 50 + e, True, 2, "2020-09-12T14:00:00Z", 2, 1) for e in (1, 2, 3)]
    data += [gw_row(1, 10, e, 40 + e, False, 1, "2020-09-12T14:00:00Z", 2, 1) for e in (4, 5)]
    data += [gw_row(2, 20, e, 60, False, 2, "2020-09-19T11:30:00Z", 0, 0) for e in (1, 2, 3)]
    data += [gw_row(2, 20, e, 45, True, 1, "2020-09-19T11:30:00Z", 0, 0) for e in (4, 5)]
    return player_rows(pd.DataFrame(data).assign(season=2020))


def test_each_players_team_comes_from_the_match(rows):
    teams = rows.drop_duplicates(["GW", "element"]).set_index(["GW", "element"])["team_id"]
    assert teams[(1, 1)] == 1 and teams[(1, 4)] == 2      # home players are team 1 in gameweek 1
    assert teams[(2, 1)] == 1 and teams[(2, 4)] == 2      # ...and still team 1 when playing away
    assert rows["price"].max() == 6.0                      # prices are stored in tenths of a million


def test_teams_link_to_our_names_by_date_and_score(rows):
    ours = pd.DataFrame({"date": ["2020-09-12"], "home": ["Arsenal"], "away": ["Fulham"],
                         "home_score": [2], "away_score": [1]})
    assert team_mapping(rows, ours) == {(2020, 1): "Arsenal", (2020, 2): "Fulham"}


def test_prices_count_only_from_when_they_were_known(rows):
    mapping = {(2020, 1): "Arsenal", (2020, 2): "Fulham"}
    gw = gameweek_values(rows, mapping, top=2)
    known = gw.set_index(["GW", "team"])["known_from"]
    assert known[(1, "Arsenal")] == pd.Timestamp("2020-07-01")    # gameweek 1 prices: from the summer
    assert known[(2, "Arsenal")] == pd.Timestamp("2020-09-19")    # later gameweeks: from first kickoff

    weekly = weekly_values(gw, pd.DatetimeIndex(["2020-09-14", "2020-09-21"])).set_index(["week", "team"])["value"]
    assert weekly[(pd.Timestamp("2020-09-14"), "Arsenal")] == pytest.approx(5.3 + 5.2)  # gameweek 1 prices
    assert weekly[(pd.Timestamp("2020-09-21"), "Arsenal")] == pytest.approx(6.0 + 6.0)  # gameweek 2 prices


def test_old_values_are_not_used(rows):
    gw = gameweek_values(rows, {(2020, 1): "Arsenal", (2020, 2): "Fulham"})
    weekly = weekly_values(gw, pd.DatetimeIndex(["2021-06-07"]))   # over 120 days after the last gameweek
    assert weekly.empty


def test_starting_prices_stand_for_the_whole_season(rows):
    from import_fpl_values import PRESEASON_STALE_DAYS
    gw = gameweek_values(rows, {(2020, 1): "Arsenal", (2020, 2): "Fulham"}, top=2)
    weeks = pd.DatetimeIndex(["2020-09-21", "2021-04-05"])
    weekly = weekly_values(gw[gw["GW"] == 1], weeks, PRESEASON_STALE_DAYS).set_index(["week", "team"])["value"]
    # Even after gameweek 2's price rise, and late in the season, the starting prices are used.
    assert weekly[(pd.Timestamp("2020-09-21"), "Arsenal")] == pytest.approx(5.3 + 5.2)
    assert weekly[(pd.Timestamp("2021-04-05"), "Arsenal")] == pytest.approx(5.3 + 5.2)
