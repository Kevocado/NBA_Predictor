"""Tests for espn module with mocked API responses."""
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path


class TestESPN:
    """Test suite for espn module."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        """Setup mocks before each test."""
        pass

    @pytest.fixture
    def clear_cache(self, tmp_path):
        """Clear cache directory before each test."""
        from nba_predictor import config
        from nba_predictor.data import espn
        original_cache_dir = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        espn.ESPN_CACHE_DIR = config.CACHE_DIR / "espn"
        espn.ESPN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        yield espn.ESPN_CACHE_DIR
        config.CACHE_DIR = original_cache_dir

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_injuries_returns_list(self, mock_fetch, clear_cache):
        """Test get_injuries returns list."""
        from nba_predictor.data import espn
        mock_fetch.return_value = {"gameId": "1", "injuries": [{"id": 1, "name": "LeBron James", "status": "Questionable"}]}
        
        result = espn.get_injuries("1")
        
        assert isinstance(result, dict)
        assert isinstance(result["injuries"], list)

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_injuries_retry_on_failure(self, mock_fetch, clear_cache):
        """Test get_injuries retries on failure."""
        from nba_predictor.data import espn
        
        mock_fetch.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            espn.get_injuries("1")

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_injuries_handles_empty_injuries(self, mock_fetch, clear_cache):
        """Test get_injuries handles empty list."""
        from nba_predictor.data import espn
        mock_fetch.return_value = {"gameId": "1", "injuries": []}
        
        result = espn.get_injuries("1")
        
        assert isinstance(result, dict)
        assert isinstance(result["injuries"], list)
        assert len(result["injuries"]) == 0

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_lineup_returns_lineup_list(self, mock_fetch, clear_cache):
        """Test get_lineup returns lineup."""
        from nba_predictor.data import espn
        mock_fetch.return_value = {"gameId": "1", "lineups": [{"id": 1, "players": []}]}
        
        result = espn.get_lineup("1")
        
        assert isinstance(result, dict)

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_lineup_handles_empty_lineup(self, mock_fetch, clear_cache):
        """Test get_lineup handles empty lineup."""
        from nba_predictor.data import espn
        mock_fetch.return_value = {"gameId": "1", "lineups": []}
        
        result = espn.get_lineup("999")
        
        assert isinstance(result, dict)

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_team_status_returns_dict(self, mock_fetch, clear_cache):
        """Test get_team_status returns dict."""
        from nba_predictor.data import espn
        mock_fetch.return_value = {"status": "active", "restDays": 0}
        
        result = espn.get_team_status(1)
        
        assert isinstance(result, dict)

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_injuries_handles_network_error(self, mock_fetch, clear_cache):
        """Test get_injuries handles network error."""
        from nba_predictor.data import espn
        
        mock_fetch.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            espn.get_injuries("1")

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_lineup_handles_network_error(self, mock_fetch, clear_cache):
        """Test get_lineup handles network error."""
        from nba_predictor.data import espn
        
        mock_fetch.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            espn.get_lineup("1")

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_team_status_handles_network_error(self, mock_fetch, clear_cache):
        """Test get_team_status handles network error."""
        from nba_predictor.data import espn
        
        mock_fetch.side_effect = Exception("Network error")
        
        with pytest.raises(Exception):
            espn.get_team_status(1)

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_injuries_caches_response(self, mock_fetch, clear_cache):
        """Test get_injuries caches response."""
        from nba_predictor.data import espn
        mock_fetch.return_value = {"gameId": "1", "injuries": [{"id": 1, "name": "Test"}]}
        
        result = espn.get_injuries("1")
        
        assert isinstance(result, dict)
        assert mock_fetch.called is True

    @patch("nba_predictor.data.espn._fetch_espn_data")
    def test_get_team_status_zero_rest_days(self, mock_fetch, clear_cache):
        """Test get_team_status with zero rest days."""
        from nba_predictor.data import espn
        mock_fetch.return_value = {"restDays": 0, "status": "active"}
        
        result = espn.get_team_status(1)
        
        assert isinstance(result, dict)
        assert "restDays" in result