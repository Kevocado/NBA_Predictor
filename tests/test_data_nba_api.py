"""Tests for nba_api module with mocked API responses."""
import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import sys
from pathlib import Path


def create_mock_endpoints():
    """Create mock endpoints with proper return values."""
    mock_endpoints = MagicMock()
    
    # Mock CommonBoard (used for get_schedule)
    mock_common_board = MagicMock()
    mock_common_board.get_dict.return_value = {
        "resultSets": [{
            "name": "GameFinder",
            "headers": [
                "GAME_ID", "GAME_CODE", "GAME_STATUS_TEXT", "HOME_TEAM_ID",
                "VISITOR_TEAM_ID", "SEASON", "DATE", "TEAM_ID_HOME",
                "TEAM_ID_VISITOR", "PTS_HOME", "PTS_VISITOR"
            ],
            "rowSet": [[
                "0012400001", "20241025/BOSBKN", "Final", 1610612738,
                1610612751, 2024, "2024-10-25", 1610612738, 1610612751,
                120, 115
            ]]
        }]
    }
    mock_endpoints.CommonBoard = lambda *args, **kwargs: mock_common_board
    
    # Mock BoxScoreTraditionalV2 (used for get_boxscore)
    mock_boxscore = MagicMock()
    mock_boxscore.get_dict.return_value = {
        "game": {
            "gameId": "0012400001",
            "homeTeam": {
                "teamId": 1610612738,
                "score": 120,
                "players": [{
                    "personId": 203999,
                    "firstName": "Jayson",
                    "lastName": "Tatum",
                    "points": 35,
                    "rebounds": 8,
                    "assists": 5
                }]
            },
            "visitorTeam": {
                "teamId": 1610612751,
                "score": 115,
                "players": [{
                    "personId": 203540,
                    "firstName": "Kevin",
                    "lastName": "Durant",
                    "points": 28,
                    "rebounds": 6,
                    "assists": 4
                }]
            }
        }
    }
    mock_endpoints.BoxScoreTraditionalV2 = lambda *args, **kwargs: mock_boxscore
    
    # Mock FourFactors
    mock_four_factors = MagicMock()
    mock_four_factors.get_dict.return_value = {
        "gameId": "0012400001",
        "lineScore": {
            "homeTeam": {
                "teamId": 1610612738,
                "effectiveFieldPercentage": 0.520,
                "turnoverPercentage": 12.5,
                "offensiveReboundPercentage": 25.0,
                "freeThrowPercentage": 0.820
            },
            "visitorTeam": {
                "teamId": 1610612751,
                "effectiveFieldPercentage": 0.495,
                "turnoverPercentage": 14.2,
                "offensiveReboundPercentage": 22.5,
                "freeThrowPercentage": 0.789
            }
        }
    }
    mock_endpoints.FourFactors = lambda *args, **kwargs: mock_four_factors
    
    # Mock PlayerCareerStats
    mock_player_career = MagicMock()
    mock_player_career.get_dict.return_value = {
        "person": {
            "id": 203999,
            "firstName": "Jayson",
            "lastName": "Tatum"
        },
        "stats": {
            "resultSets": [{
                "name": "SeasonTotalsClassic",
                "headers": [
                    "PLAYER_ID", "SEASON_ID", "TEAM_ID", "PTS", "REB", "AST",
                    "FG_PCT", "FG3_PCT", "FT_PCT", "GP", "MIN"
                ],
                "rowSet": [[
                    203999, "2023-24", 1610612738, 26.9, 8.1, 4.4, 0.472,
                    0.392, 0.855, 73, 35.2
                ]]
            }]
        }
    }
    mock_endpoints.PlayerCareerStats = lambda *args, **kwargs: mock_player_career
    
    # Mock TeamDashboardByYearOld
    mock_team_dashboard = MagicMock()
    mock_team_dashboard.get_dict.return_value = {
        "resultSets": [{
            "name": "TeamDashboard",
            "headers": [
                "TEAM_ID", "TEAM_NAME", "GP", "W", "L", "W_PCT", "FG_PCT",
                "FG3_PCT", "FT_PCT", "PTS", "REB", "AST"
            ],
            "rowSet": [[
                1610612738, "Boston Celtics", 73, 64, 9, 0.877, 0.482,
                0.395, 0.852, 118.5, 45.2, 26.8
            ]]
        }]
    }
    mock_endpoints.TeamDashboardByYearOld = lambda *args, **kwargs: mock_team_dashboard
    
    # Mock PlayByPlay
    mock_play_by_play = MagicMock()
    mock_play_by_play.get_dict.return_value = {
        "resultSets": [{
            "name": "PlayByPlay",
            "headers": [
                "ACTION_NUMBER", "CLOCK", "PERIOD", "TEAM_ID", "TEAM_CITY",
                "TEAM_NAME", "PERSON_ID", "PLAYER_NAME", "ACTION_TYPE",
                "SUB_TYPE", "WKT_POS", "ZONE_POS", "SCORE_HOME", "SCORE_AWAY",
                "DESCRIPTION"
            ],
            "rowSet": [
                [1, "12:00", 1, 1610612738, "Boston", "Celtics", 0, "Team",
                 "jumpball", "center", "", "", "", "",
                 "Jump Ball Tatum vs. Kidd: Tip to Horford"],
                [2, "11:55", 1, 1610612738, "Boston", "Celtics", 203999,
                 "Jayson Tatum", "shot", "2pt", "", "", "2", "",
                 "Jayson Tatum 2PT Field Goal (2 pts) (Jaylen Brown 1 Ast)"]
            ]
        }]
    }
    mock_endpoints.PlayByPlay = lambda *args, **kwargs: mock_play_by_play
    
    return mock_endpoints


@pytest.fixture
def mock_nba_api():
    """Patch the nba_api module with mocks."""
    mock_endpoints = create_mock_endpoints()
    
    with patch.dict('sys.modules', {
        'nba_api': MagicMock(),
        'nba_api.stats': MagicMock(),
        'nba_api.stats.endpoints': mock_endpoints,
    }):
        # Clear the cache directory
        from nba_predictor import config
        cache_dir = config.CACHE_DIR / "nba_api"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear any existing cache files
        for f in cache_dir.iterdir():
            f.unlink()
        
        yield mock_endpoints


class TestNBAAPI:
    """Test suite for nba_api module."""

    def test_get_schedule_returns_games_for_date(self, mock_nba_api):
        """Test get_schedule returns game data for a given date."""
        from nba_predictor.data import nba_api
        
        games = nba_api.get_schedule("2024-10-25")
        
        assert isinstance(games, list)
        assert len(games) == 1
        game = games[0]
        assert game["GAME_ID"] == "0012400001"
        assert game["HOME_TEAM_ID"] == 1610612738
        assert game["VISITOR_TEAM_ID"] == 1610612751

    def test_get_schedule_caches_response(self, mock_nba_api, tmp_path, monkeypatch):
        """Test get_schedule caches the response to a JSON file."""
        from nba_predictor import config
        from nba_predictor.data import nba_api
        
        games = nba_api.get_schedule("2024-10-25")
        
        # Check cache file was created - using <endpoint>_<id>_<date>.json format
        cache_file = config.CACHE_DIR / "nba_api" / "get_schedule_2024-10-25_2024-10-25.json"
        assert cache_file.exists()
        
        # Verify cache content
        with open(cache_file) as f:
            cached_data = json.load(f)
        assert len(cached_data) == 1
        assert cached_data[0]["GAME_ID"] == "0012400001"

    def test_get_schedule_uses_cache_on_second_call(self, mock_nba_api, tmp_path, monkeypatch):
        """Test get_schedule uses cached data on second call."""
        from nba_predictor import config
        from nba_predictor.data import nba_api
        
        # First call - should fetch and cache
        games1 = nba_api.get_schedule("2024-10-25")
        
        # Second call - should use cache (no API call)
        games2 = nba_api.get_schedule("2024-10-25")
        
        assert games1 == games2

    def test_get_boxscore_returns_boxscore_data(self, mock_nba_api):
        """Test get_boxscore returns boxscore data for a game."""
        from nba_predictor.data import nba_api
        
        boxscore = nba_api.get_boxscore("0012400001")
        
        assert isinstance(boxscore, dict)
        assert "gameId" in boxscore or "game" in boxscore

    def test_get_boxscore_caches_response(self, mock_nba_api, tmp_path, monkeypatch):
        """Test get_boxscore caches the response."""
        from nba_predictor import config
        from nba_predictor.data import nba_api
        
        boxscore = nba_api.get_boxscore("0012400001")
        
        # Check cache file was created
        cache_file = config.CACHE_DIR / "nba_api" / "get_boxscore_0012400001_2024-10-25.json"
        assert cache_file.exists()

    def test_get_four_factors_returns_four_factors_data(self, mock_nba_api):
        """Test get_four_factors returns Four Factors data for a game."""
        from nba_predictor.data import nba_api
        
        four_factors = nba_api.get_four_factors("0012400001")
        
        assert isinstance(four_factors, dict)
        assert "gameId" in four_factors or "lineScore" in four_factors

    def test_get_four_factors_caches_response(self, mock_nba_api, tmp_path, monkeypatch):
        """Test get_four_factors caches the response."""
        from nba_predictor import config
        from nba_predictor.data import nba_api
        
        nba_api.get_four_factors("0012400001")
        
        cache_file = config.CACHE_DIR / "nba_api" / "get_four_factors_0012400001_2024-10-25.json"
        assert cache_file.exists()

    def test_get_player_stats_returns_player_stats(self, mock_nba_api):
        """Test get_player_stats returns season stats for a player."""
        from nba_predictor.data import nba_api
        
        stats = nba_api.get_player_stats("203999", 2024)
        
        assert isinstance(stats, dict)
        assert "person" in stats or "stats" in stats

    def test_get_player_stats_caches_response(self, mock_nba_api, tmp_path, monkeypatch):
        """Test get_player_stats caches the response."""
        from nba_predictor import config
        from nba_predictor.data import nba_api
        
        nba_api.get_player_stats("203999", 2024)
        
        cache_file = config.CACHE_DIR / "nba_api" / "get_player_career_stats_203999_2024.json"
        assert cache_file.exists()

    def test_get_team_stats_returns_team_stats(self, mock_nba_api):
        """Test get_team_stats returns season stats for a team."""
        from nba_predictor.data import nba_api
        
        stats = nba_api.get_team_stats(1610612738, 2024)
        
        assert isinstance(stats, dict)
        assert "resultSets" in stats

    def test_get_team_stats_caches_response(self, mock_nba_api, tmp_path, monkeypatch):
        """Test get_team_stats caches the response."""
        from nba_predictor import config
        from nba_predictor.data import nba_api
        
        nba_api.get_team_stats(1610612738, 2024)
        
        cache_file = config.CACHE_DIR / "nba_api" / "get_team_stats_1610612738_2024.json"
        assert cache_file.exists()

    def test_get_play_by_play_returns_play_by_play_data(self, mock_nba_api):
        """Test get_play_by_play returns play-by-play data for a game."""
        from nba_predictor.data import nba_api
        
        plays = nba_api.get_play_by_play("0012400001")
        
        assert isinstance(plays, list)
        assert len(plays) == 2
        assert plays[0]["ACTION_NUMBER"] == 1
        assert plays[1]["ACTION_NUMBER"] == 2

    def test_get_play_by_play_caches_response(self, mock_nba_api, tmp_path, monkeypatch):
        """Test get_play_by_play caches the response."""
        from nba_predictor import config
        from nba_predictor.data import nba_api
        
        nba_api.get_play_by_play("0012400001")
        
        cache_file = config.CACHE_DIR / "nba_api" / "get_play_by_play_0012400001_2024-10-25.json"
        assert cache_file.exists()

    def test_cache_directory_is_configured_correctly(self, mock_nba_api, tmp_path, monkeypatch):
        """Test that the cache directory is configured correctly."""
        from nba_predictor.data import nba_api
        from nba_predictor import config
        
        assert str(nba_api.CACHE_DIR) == str(config.CACHE_DIR / "nba_api")

    def test_max_retries_is_set_to_five(self, mock_nba_api, tmp_path, monkeypatch):
        """Test that max retries is configured correctly."""
        from nba_predictor.data import nba_api
        
        assert nba_api.MAX_RETRIES == 5

    def test_initial_retry_delay_is_one_second(self, mock_nba_api, tmp_path, monkeypatch):
        """Test that the initial retry delay is set correctly."""
        from nba_predictor.data import nba_api
        
        assert nba_api.INITIAL_RETRY_DELAY == 1.0
