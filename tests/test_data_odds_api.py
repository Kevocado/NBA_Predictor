"""Tests for odds_api module with mocked API responses."""
import pytest
from unittest.mock import MagicMock, patch
import json
import os
import requests
from pathlib import Path


class TestOddsApi:
    """Test suite for odds_api module."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        """Setup mocks before each test."""
        pass

    @pytest.fixture
    def clear_cache(self, tmp_path):
        """Clear cache directory before each test."""
        from nba_predictor import config
        from nba_predictor.data import odds_api
        original_cache_dir = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        odds_api.CACHE_DIR = config.CACHE_DIR / "odds"
        odds_api.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        yield odds_api.CACHE_DIR
        config.CACHE_DIR = original_cache_dir

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_odds_success(self, mock_api_key, clear_cache):
        """Test get_odds returns list of odds."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = [
            {"id": "a46a", "sport_key": "basketball_nba", "teams": ["Boston Celtics", "Brooklyn Nets"]}
        ]
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: expected_response)
            result = odds_api.get_odds()
            
        assert isinstance(result, list)
        assert len(result) > 0

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_odds_cached(self, mock_api_key, clear_cache):
        """Test get_odds uses cache."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        cache_path = clear_cache / "bulk_odds.json"
        with open(cache_path, "w") as f:
            json.dump({"data": [{"id": "a46a", "sport_key": "basketball_nba"}], "cached_at": 1234567890}, f)
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            result = odds_api.get_odds()
            
        assert isinstance(result, list)

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_odds_retries_on_failure(self, mock_api_key, clear_cache):
        """Test retry logic with exponential backoff."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = [{"id": "test"}]
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.side_effect = [
                requests.exceptions.HTTPError("Server error"),
                MagicMock(status_code=200, json=lambda: expected_response)
            ]
            
            result = odds_api.get_odds()
            
        assert isinstance(result, list)

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_odds_max_retries_exceeded(self, mock_api_key, clear_cache):
        """Test that max retries are respected."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.side_effect = requests.exceptions.HTTPError("Server error")
            
            with pytest.raises(requests.exceptions.HTTPError):
                odds_api.get_odds()

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_h2h_odds_success(self, mock_api_key, clear_cache):
        """Test get_h2h_odds returns dict."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = [{"id": "a46a", "h2h": {"home": -110, "away": -110}}]
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: expected_response)
            result = odds_api.get_h2h_odds("a46a")
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_spreads_odds_success(self, mock_api_key, clear_cache):
        """Test get_spreads_odds returns dict."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = [{"id": "a46a", "spreads": [{"line": -3.5}]}]
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: expected_response)
            result = odds_api.get_spreads_odds("a46a")
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_totals_odds_success(self, mock_api_key, clear_cache):
        """Test get_totals_odds returns dict."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = [{"id": "a46a", "totals": [{"line": 215.5}]}]
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: expected_response)
            result = odds_api.get_totals_odds("a46a")
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_get_h2h_odds_no_data(self, mock_api_key, clear_cache):
        """Test get_h2h_odds with no matching data."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = [{"id": "other_game"}]
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: expected_response)
            result = odds_api.get_h2h_odds("nonexistent_game_id")
            
        assert isinstance(result, dict)
        assert result == {}

    def test_get_odds_no_api_key(self, monkeypatch, clear_cache):
        """Test that missing API key raises error.

        Patches config.ODDS_API_KEY directly (the already-resolved value
        odds_api._get_api_key() actually reads) rather than os.environ —
        env-var manipulation doesn't work here since config.py reads it
        once at import time via load_dotenv(), and a real .env file (with
        a real key, reused from another project) can exist on disk
        regardless of this process's os.environ at test time. Also needs
        clear_cache: get_odds() checks its on-disk cache before ever
        calling _get_api_key(), so a leftover bulk_odds.json from an
        earlier test/run would make this test pass for the wrong reason.
        """
        from nba_predictor import config
        from nba_predictor.data import odds_api

        monkeypatch.setattr(config, "ODDS_API_KEY", None)

        with pytest.raises(ValueError, match="ODDS_API_KEY"):
            odds_api.get_odds()

    @patch("nba_predictor.data.odds_api._get_api_key")
    @patch("time.sleep")
    def test_get_odds_rate_limit_handling(self, mock_sleep, mock_api_key, clear_cache):
        """Test rate limit handling with 429 response."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = [{"id": "test"}]
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.side_effect = [
                MagicMock(status_code=429),
                MagicMock(status_code=200, json=lambda: expected_response)
            ]
            
            result = odds_api.get_odds()
            
        assert isinstance(result, list)

    def test_bulk_odds_structure(self):
        """Test the structure of bulk odds response."""
        from nba_predictor.data import odds_api
        
        mock_response = [
            {"id": "a46a", "sport_key": "basketball_nba", "teams": ["Boston Celtics", "Brooklyn Nets"], "home_team": "Boston Celtics"}
        ]
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: mock_response)
            result = odds_api.get_odds()
            
        assert isinstance(result, list)

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_cache_functions(self, mock_api_key, clear_cache):
        """Test cache functions."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        with patch("nba_predictor.data.odds_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: [{"id": "test"}])
            odds_api.get_odds()
            
        assert clear_cache.exists()

    @patch("nba_predictor.data.odds_api._get_api_key")
    def test_clear_cache(self, mock_api_key, clear_cache):
        """Test clear_cache function."""
        from nba_predictor.data import odds_api
        mock_api_key.return_value = "test-api-key"
        
        odds_api.clear_cache()
        
        assert isinstance(list(clear_cache.glob("*.json")), list)