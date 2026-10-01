"""The promoted-teams check."""

import pandas as pd

from promoted_teams import promoted_teams


def test_promoted_teams_are_new_to_the_league_that_season():
    games = pd.DataFrame({
        "season": [2022, 2022, 2023, 2023],
        "home":   ["A", "B", "A", "C"],
        "away":   ["B", "A", "C", "A"],
    })
    assert promoted_teams(games) == {(2023, "C")}   # the first season has no previous season to compare
