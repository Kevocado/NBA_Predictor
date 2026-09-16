from unittest.mock import patch

import pytest


@pytest.fixture
def clear_cache(tmp_path, monkeypatch):
    from nba_predictor.data import espn

    monkeypatch.setattr(espn, "ESPN_CACHE_DIR", tmp_path / "espn")
    return espn.ESPN_CACHE_DIR


def test_normalize_abbreviation_maps_known_mismatches():
    from nba_predictor.data.espn import normalize_abbreviation

    assert normalize_abbreviation("GS") == "GSW"
    assert normalize_abbreviation("UTAH") == "UTA"
    assert normalize_abbreviation("NY") == "NYK"


def test_normalize_abbreviation_passes_through_matching_ones():
    from nba_predictor.data.espn import normalize_abbreviation

    assert normalize_abbreviation("BOS") == "BOS"


@patch("nba_predictor.data.espn._fetch_json")
def test_get_scoreboard_parses_completed_game(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {
        "events": [
            {
                "id": "401810448",
                "competitions": [
                    {
                        "status": {"type": {"completed": True}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "DAL"}, "score": "138"},
                            {"homeAway": "away", "team": {"abbreviation": "UTAH"}, "score": "120"},
                        ],
                    }
                ],
            }
        ]
    }

    games = espn.get_scoreboard("2026-01-17")

    assert len(games) == 1
    game = games[0]
    assert game["game_id"] == "401810448"
    assert game["home_team"] == "DAL"
    assert game["away_team"] == "UTA"
    assert game["completed"] is True
    assert game["home_pts"] == 138
    assert game["away_pts"] == 120


@patch("nba_predictor.data.espn._fetch_json")
def test_get_scoreboard_handles_upcoming_game_without_score(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {
        "events": [
            {
                "id": "1",
                "competitions": [
                    {
                        "status": {"type": {"completed": False}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "BOS"}},
                            {"homeAway": "away", "team": {"abbreviation": "MIA"}},
                        ],
                    }
                ],
            }
        ]
    }

    games = espn.get_scoreboard("2026-11-01")

    assert games[0]["completed"] is False
    assert games[0]["home_pts"] is None


@patch("nba_predictor.data.espn._fetch_json")
def test_get_scoreboard_caches_across_calls(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {"events": []}

    espn.get_scoreboard("2026-01-17")
    espn.get_scoreboard("2026-01-17")

    assert mock_fetch.call_count == 1


@patch("nba_predictor.data.espn._fetch_json")
def test_get_boxscore_parses_four_factors_inputs(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {
        "boxscore": {
            "teams": [
                {
                    "team": {"abbreviation": "UTAH"},
                    "statistics": [
                        {"name": "fieldGoalsMade-fieldGoalsAttempted", "displayValue": "48-89"},
                        {"name": "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "displayValue": "10-31"},
                        {"name": "freeThrowsMade-freeThrowsAttempted", "displayValue": "14-20"},
                        {"name": "offensiveRebounds", "displayValue": "11"},
                        {"name": "defensiveRebounds", "displayValue": "30"},
                        {"name": "turnovers", "displayValue": "15"},
                    ],
                }
            ]
        }
    }

    box = espn.get_boxscore("401810448")

    assert box["UTA"]["fgm"] == 48.0
    assert box["UTA"]["fga"] == 89.0
    assert box["UTA"]["fg3m"] == 10.0
    assert box["UTA"]["ftm"] == 14.0
    assert box["UTA"]["fta"] == 20.0
    assert box["UTA"]["oreb"] == 11.0
    assert box["UTA"]["dreb"] == 30.0
    assert box["UTA"]["tov"] == 15.0


@patch("nba_predictor.data.espn._fetch_json")
def test_get_injuries_maps_team_display_name_to_abbreviation(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {
        "injuries": [
            {
                "displayName": "Boston Celtics",
                "injuries": [{"athlete": {"displayName": "Jayson Tatum"}, "status": "Day-To-Day"}],
            }
        ]
    }

    injuries = espn.get_injuries()

    assert injuries == [{"team": "BOS", "player_name": "Jayson Tatum", "status": "Day-To-Day"}]


@patch("nba_predictor.data.espn._fetch_json")
def test_get_injuries_retries_and_raises_on_persistent_failure(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.side_effect = Exception("network error")

    with pytest.raises(Exception):
        espn.get_injuries()
