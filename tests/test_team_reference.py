import pytest


def test_has_exactly_thirty_teams():
    from nba_predictor.data.team_reference import TEAMS

    assert len(TEAMS) == 30


def test_abbreviations_are_unique():
    from nba_predictor.data.team_reference import TEAMS

    abbrs = [t.abbreviation for t in TEAMS]
    assert len(abbrs) == len(set(abbrs))


def test_conferences_are_east_or_west_15_each():
    from nba_predictor.data.team_reference import TEAMS

    conferences = [t.conference for t in TEAMS]
    assert set(conferences) == {"East", "West"}
    assert conferences.count("East") == 15
    assert conferences.count("West") == 15


def test_divisions_are_six_of_five_teams_each():
    from nba_predictor.data.team_reference import TEAMS

    from collections import Counter

    counts = Counter(t.division for t in TEAMS)
    assert len(counts) == 6
    assert all(count == 5 for count in counts.values())


def test_only_denver_has_nonzero_altitude():
    from nba_predictor.data.team_reference import TEAMS

    high_altitude = [t for t in TEAMS if t.altitude_ft > 1000]
    assert len(high_altitude) == 1
    assert high_altitude[0].abbreviation == "DEN"


def test_get_team_returns_matching_team():
    from nba_predictor.data.team_reference import get_team

    team = get_team("BOS")
    assert team.name == "Boston Celtics"
    assert team.conference == "East"
    assert team.division == "Atlantic"


def test_get_team_raises_for_unknown_abbreviation():
    from nba_predictor.data.team_reference import get_team

    with pytest.raises(KeyError):
        get_team("ZZZ")
