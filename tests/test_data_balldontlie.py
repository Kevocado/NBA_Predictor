"""Tests for balldontlie module with mocked API responses."""
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path


def _make_mock_response(data):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = data
    return mock_resp


class TestBalldontlie:
    """Test suite for balldontlie module."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        """Setup mocks before each test."""
        pass

    @pytest.fixture
    def clear_cache(self, tmp_path):
        """Clear cache directory before each test."""
        from nba_predictor import config
        from nba_predictor.data import balldontlie
        original_cache_dir = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        balldontlie._CACHE_DIR = config.CACHE_DIR / "balldontlie"
        balldontlie._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        yield balldontlie._CACHE_DIR
        config.CACHE_DIR = original_cache_dir

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_schedule_success(self, mock_api_key, clear_cache):
        """Test get_schedule returns games for a date."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {
            "data": [{"id": 1, "game_id": "0012400001", "date": "2024-10-25"}]
        }
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            games = balldontlie.get_schedule("2024-10-25")
            
        assert isinstance(games, list)

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_schedule_empty(self, mock_api_key, clear_cache):
        """Test get_schedule with empty response."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {"data": []}
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            games = balldontlie.get_schedule("2024-10-25")
            
        assert isinstance(games, list)

    @patch("nba_predictor.data.balldontlie._get_api_key")
    @patch("time.sleep")
    def test_get_schedule_rate_limit(self, mock_sleep, mock_api_key, clear_cache):
        """Test get_schedule handles rate limiting."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_request.side_effect = [mock_response] * 5
            
            with pytest.raises(Exception):
                balldontlie.get_schedule("2024-10-25")

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_boxscore_success(self, mock_api_key, clear_cache):
        """Test get_boxscore returns boxscore data."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {
            "data": {"id": "0012400001", "home_team": {"id": 1610612738, "score": 120}}
        }
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            result = balldontlie.get_boxscore(1)
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_boxscore_not_found(self, mock_api_key, clear_cache):
        """Test get_boxscore when game not found."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = {"data": None}
            result = balldontlie.get_boxscore(999999)
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_stats_success(self, mock_api_key, clear_cache):
        """Test get_player_stats returns player stats."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {
            "data": [{"id": 203999, "first_name": "Jayson", "pts_per_game": 26.9}]
        }
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            result = balldontlie.get_player_stats(203999, 2023)
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_stats_no_data(self, mock_api_key, clear_cache):
        """Test get_player_stats with no data."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {"data": []}
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            result = balldontlie.get_player_stats(999999, 2023)
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_team_stats_success(self, mock_api_key, clear_cache):
        """Test get_team_stats returns team stats."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {
            "data": [{"team_id": 1610612738, "season": 2023, "wins": 64}]
        }
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            result = balldontlie.get_team_stats(1610612738, 2023)
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_team_stats_not_found(self, mock_api_key, clear_cache):
        """Test team stats when team not found."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {"data": []}
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            result = balldontlie.get_team_stats(999999, 2023)
            
        assert isinstance(result, dict)
        assert result == {}

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_game_log_success(self, mock_api_key, clear_cache):
        """Test get_player_game_log returns game log."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {
            "data": [{"id": 1, "player_id": 203999, "pts": 30}, {"id": 2, "player_id": 203999, "pts": 28}]
        }
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            result = balldontlie.get_player_game_log(203999, 2023)
            
        assert isinstance(result, list)
        assert len(result) == 2

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_game_log_empty(self, mock_api_key, clear_cache):
        """Test game log with no games."""
        from nba_predictor.data import balldontlie
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {"data": []}
        
        with patch("nba_predictor.data.balldontlie._make_request") as mock_request:
            mock_request.return_value = expected_response
            result = balldontlie.get_player_game_log(203999, 2023)
            
        assert isinstance(result, list)
        assert len(result) == 0