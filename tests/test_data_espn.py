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
                "date": "2026-01-18T01:30Z",
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
    assert game["tip_off"] == "2026-01-18T01:30Z"


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
def test_get_scoreboard_skips_non_standard_exhibition_events(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {
        "events": [
            {
                "id": "401812487",
                "name": "Melbourne Pnx at New Orleans Pelicans",
                "competitions": [
                    {
                        "status": {"type": {"completed": True}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "NOP"}, "score": "100"},
                            {"homeAway": "away"},
                        ],
                    }
                ],
            },
            {
                "id": "1",
                "competitions": [
                    {
                        "status": {"type": {"completed": True}},
                        "competitors": [
                            {"homeAway": "home", "team": {"abbreviation": "BOS"}, "score": "110"},
                            {"homeAway": "away", "team": {"abbreviation": "MIA"}, "score": "100"},
                        ],
                    }
                ],
            },
        ]
    }

    games = espn.get_scoreboard("2025-10-04")

    assert len(games) == 1
    assert games[0]["game_id"] == "1"


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
def test_get_player_boxscore_parses_played_athletes(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "UTAH"},
                    "statistics": [
                        {
                            "labels": ["MIN", "PTS", "FG", "3PT", "FT", "REB", "AST"],
                            "athletes": [
                                {
                                    "didNotPlay": False,
                                    "stats": ["29", "13", "6-11", "1-4", "0-0", "12", "5"],
                                    "athlete": {"id": "1", "displayName": "Kyle Filipowski", "position": {"abbreviation": "F"}},
                                },
                                {
                                    "didNotPlay": True,
                                    "stats": [],
                                    "athlete": {"id": "2", "displayName": "Kevin Love", "position": {"abbreviation": "F"}},
                                },
                            ],
                        }
                    ],
                }
            ]
        }
    }

    rows = espn.get_player_boxscore("401810448")

    assert len(rows) == 1
    assert rows[0]["player_name"] == "Kyle Filipowski"
    assert rows[0]["team"] == "UTA"
    assert rows[0]["points"] == 13.0
    assert rows[0]["rebounds"] == 12.0
    assert rows[0]["fg_made_attempted"] == "6-11"


@patch("nba_predictor.data.espn._fetch_json")
def test_get_player_boxscore_caches_across_calls(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = {"boxscore": {"players": []}}

    espn.get_player_boxscore("1")
    espn.get_player_boxscore("1")

    assert mock_fetch.call_count == 1


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

    # player_id is blank here: this fixture has no athlete links to read an id
    # from, and an unreadable id must resolve to nobody rather than guess.
    assert injuries == [
        {
            "team": "BOS",
            "player_id": "",
            "player_name": "Jayson Tatum",
            "status": "Day-To-Day",
            "dated": "",
        }
    ]


@patch("nba_predictor.data.espn._fetch_json")
def test_get_injuries_retries_and_raises_on_persistent_failure(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.side_effect = Exception("network error")

    with pytest.raises(Exception):
        espn.get_injuries()


@patch("nba_predictor.data.espn._fetch_json")
def test_get_scoreboard_carries_espns_season_type(mock_fetch, clear_cache):
    """1 = preseason, 2 = regular season, 3 = postseason: a preseason minute is not a regular-season minute."""
    from nba_predictor.data import espn

    event = lambda eid, season_type: {
        "id": eid, "date": "2026-10-10T23:00Z", "season": {"type": season_type},
        "competitions": [{"status": {"type": {"completed": False}}, "competitors": [
            {"homeAway": "home", "team": {"abbreviation": "BOS"}}, {"homeAway": "away", "team": {"abbreviation": "NYK"}}]}],
    }
    mock_fetch.return_value = {"events": [event("1", 1), event("2", 2)]}
    assert [g["season_type"] for g in espn.get_scoreboard("2026-10-10")] == [1, 2]


@patch("nba_predictor.data.espn._fetch_json")
def test_a_cached_scoreboard_without_season_type_is_refetched(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    espn._save_cache("scoreboard", "2026-10-10", {"games": [{"game_id": "1", "game_date": "2026-10-10", "home_team": "BOS", "away_team": "NYK", "completed": False}]})
    mock_fetch.return_value = {"events": [{
        "id": "1", "date": "2026-10-10T23:00Z", "season": {"type": 1},
        "competitions": [{"status": {"type": {"completed": False}}, "competitors": [
            {"homeAway": "home", "team": {"abbreviation": "BOS"}}, {"homeAway": "away", "team": {"abbreviation": "NYK"}}]}]}]}
    assert espn.get_scoreboard("2026-10-10")[0]["season_type"] == 1


def _board(completed):
    return {"events": [{"id": "1", "date": "2026-10-20T19:00Z", "season": {"type": 2}, "competitions": [{
        "status": {"type": {"completed": completed}},
        "competitors": [{"homeAway": "home", "team": {"abbreviation": "DET"}, "score": "121"},
                        {"homeAway": "away", "team": {"abbreviation": "BOS"}, "score": "119"}]}]}]}


@patch("nba_predictor.data.espn._fetch_json")
def test_a_day_cached_before_its_games_finished_is_refetched(mock_fetch, clear_cache):
    from nba_predictor.data import espn

    mock_fetch.return_value = _board(False)
    assert espn.get_scoreboard("2026-10-20")[0]["completed"] is False
    mock_fetch.return_value = _board(True)
    assert espn.get_scoreboard("2026-10-20")[0]["completed"] is True      # not stuck on the cached "awaiting result"
    assert espn.get_scoreboard("2026-10-20")[0]["home_pts"] == 121
    assert mock_fetch.call_count == 2                                      # the finished day is now cached and final
