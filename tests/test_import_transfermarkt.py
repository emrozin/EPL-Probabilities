"""Squad values from Transfermarkt data: only information available on each date is used."""

import pandas as pd
import pytest

from import_transfermarkt import club_mapping, squad_values

M = 1_000_000


def test_clubs_are_linked_by_matching_games():
    tm = pd.DataFrame({"competition_id": ["GB1", "GB1", "ES1"], "date": ["2023-08-12", "2023-08-19", "2023-08-12"],
                       "home_club_id": [11, 985, 418], "away_club_id": [985, 11, 131],
                       "home_club_goals": [2, 1, 2], "away_club_goals": [1, 0, 1]})
    ours = pd.DataFrame({"date": ["2023-08-12", "2023-08-19"], "home": ["Arsenal", "Man United"],
                         "away": ["Man United", "Arsenal"], "home_score": [2, 1], "away_score": [1, 0]})
    assert club_mapping(tm, ours) == {11: "Arsenal", 985: "Man United"}   # the Spanish game is ignored


@pytest.fixture
def scenario():
    transfers = pd.DataFrame([
        # player, date, to club
        (1, "2015-07-01", 1),    # joins club 1 in the summer
        (2, "2015-08-20", 1),    # joins club 1 late in August
        (3, "2014-07-01", 1),
        (3, "2015-08-10", 99),   # ...and is loaned out to a club we don't track
        (4, "2011-07-01", 1),    # long gone: no valuation for years (retired)
        (5, "2014-07-01", 1),
    ], columns=["player_id", "transfer_date", "to_club_id"])
    appearances = pd.DataFrame([(5, "2015-05-01", 1)], columns=["player_id", "date", "player_club_id"])
    valuations = pd.DataFrame([
        (1, "2015-06-01", 50 * M),
        (2, "2015-06-01", 30 * M),
        (3, "2015-06-01", 20 * M),
        (4, "2011-06-01", 99 * M),
        (5, "2015-06-01", 10 * M),
        (5, "2015-08-25", 40 * M),   # re-valued after the first Monday checked below
    ], columns=["player_id", "date", "market_value_in_eur"])
    weeks = pd.DatetimeIndex(["2015-08-17", "2015-08-31"])
    out = squad_values(transfers, appearances, valuations, {1}, weeks)
    return out.set_index("week")


def test_only_players_at_the_club_on_that_date_count(scenario):
    aug17 = scenario.loc["2015-08-17"]
    assert aug17["players"] == 2                  # players 1 and 5; 2 hasn't joined, 3 is on loan, 4 has retired
    assert aug17["value_eur"] == 50 * M + 10 * M  # player 5 at his value before the later re-valuation


def test_later_signings_and_valuations_count_from_their_dates(scenario):
    aug31 = scenario.loc["2015-08-31"]
    assert aug31["players"] == 3                  # player 2 has now joined
    assert aug31["value_eur"] == 50 * M + 30 * M + 40 * M


def test_only_the_most_valuable_players_count():
    transfers = pd.DataFrame({"player_id": [1, 2, 3], "transfer_date": ["2015-07-01"] * 3, "to_club_id": [1, 1, 1]})
    valuations = pd.DataFrame({"player_id": [1, 2, 3], "date": ["2015-06-01"] * 3,
                               "market_value_in_eur": [5 * M, 50 * M, 20 * M]})
    appearances = pd.DataFrame(columns=["player_id", "date", "player_club_id"])
    out = squad_values(transfers, appearances, valuations, {1}, pd.DatetimeIndex(["2015-08-17"]), top=2)
    assert out["value_eur"].tolist() == [70 * M]


def test_long_serving_players_count_without_recent_matches_in_the_data():
    """Regression: a player who joined years ago and plays where the dataset has no matches
    (like the Championship) still counts, as long as he's being valued."""
    transfers = pd.DataFrame({"player_id": [1], "transfer_date": ["2016-07-01"], "to_club_id": [1]})
    valuations = pd.DataFrame({"player_id": [1, 1], "date": ["2016-06-01", "2020-06-01"],
                               "market_value_in_eur": [5 * M, 12 * M]})
    appearances = pd.DataFrame(columns=["player_id", "date", "player_club_id"])
    out = squad_values(transfers, appearances, valuations, {1}, pd.DatetimeIndex(["2020-08-10"]))
    assert out["value_eur"].tolist() == [12 * M]
