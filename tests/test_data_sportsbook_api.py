"""Tests for sportsbook_api module with mocked API responses."""
import pytest
from unittest.mock import MagicMock, patch
import json
import os
from pathlib import Path


class TestSportsbookAPI:
    """Test suite for sportsbook_api module."""

    @pytest.fixture(autouse=True)
    def setup_module(self):
        """Setup mocks before each test."""
        pass

    @pytest.fixture
    def clear_cache(self, tmp_path):
        """Clear cache directory before each test."""
        from nba_predictor import config
        from nba_predictor.data import sportsbook_api
        original_cache_dir = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        sportsbook_api.CACHE_DIR = config.CACHE_DIR / "sportsbook"
        sportsbook_api.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        yield sportsbook_api.CACHE_DIR
        config.CACHE_DIR = original_cache_dir

    def test_cache_key_returns_string(self, clear_cache):
        """Test cache_key returns a string."""
        from nba_predictor.data import sportsbook_api
        
        result = sportsbook_api.cache_key("0012400001")
        
        assert isinstance(result, str)
        assert "0012400001" in result

    def test_cache_key_format(self, clear_cache):
        """Test cache_key has correct format."""
        from nba_predictor.data import sportsbook_api
        
        result = sportsbook_api.cache_key("0012400001")
        
        assert result.endswith(".json")
        assert "odds_0012400001" in result

    def test_cache_key_unique_per_game_id(self, clear_cache):
        """Test cache_key is unique per game ID."""
        from nba_predictor.data import sportsbook_api
        
        result1 = sportsbook_api.cache_key("0012400001")
        result2 = sportsbook_api.cache_key("0012400002")
        
        assert result1 != result2

    @patch("nba_predictor.data.sportsbook_api._get_api_key")
    def test_get_odds_returns_dict(self, mock_api_key, clear_cache):
        """Test get_odds returns dict with odds data."""
        from nba_predictor.data import sportsbook_api
        mock_api_key.return_value = "test-api-key"
        
        expected_response = {
            "id": "0012400001",
            "home_team": "Boston Celtics",
            "away_team": "Brooklyn Nets",
            "odds": {"h2h": {"Boston Celtics": -110, "Brooklyn Nets": -110}},
        }
        
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
            mock_fetch.return_value = expected_response
            result = sportsbook_api.get_odds("0012400001")
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.sportsbook_api._get_api_key")
    def test_get_odds_uses_cache_when_fresh(self, mock_api_key, clear_cache):
        """Test get_odds uses cache when fresh."""
        from nba_predictor.data import sportsbook_api
        mock_api_key.return_value = "test-api-key"
        
        # Pre-populate cache
        cache_path = clear_cache / "odds_0012400001.json"
        cache_data = {"id": "0012400001", "cached_at": 9999999999}  # Far future timestamp
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
        
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
            result = sportsbook_api.get_odds("0012400001")
            
        assert isinstance(result, dict)

    @patch("nba_predictor.data.sportsbook_api._get_api_key")
    def test_get_odds_refreshes_expired_cache(self, mock_api_key, clear_cache):
        """Test get_odds refreshes expired cache."""
        from nba_predictor.data import sportsbook_api
        mock_api_key.return_value = "test-api-key"
        
        # Create expired cache file (older than 6 hours)
        import time
        from datetime import datetime, timedelta
        cache_path = clear_cache / "odds_0012400001.json"
        old_time = (datetime.now() - timedelta(hours=7)).timestamp()
        expired_data = {"data": {"id": "0012400001"}, "cached_at": old_time}
        with open(cache_path, "w") as f:
            json.dump(expired_data, f)
        
        # Mock API response
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
            mock_fetch.return_value = {"id": "0012400001", "odds": {}}
            
            result = sportsbook_api.get_odds("0012400001")
            
        assert isinstance(result, dict)
        # Verify API was called for expired cache
        assert mock_fetch.called is True

    @patch("nba_predictor.data.sportsbook_api._get_api_key")
    def test_get_odds_handles_api_error(self, mock_api_key, clear_cache):
        """Test get_odds handles API error."""
        from nba_predictor.data import sportsbook_api
        mock_api_key.return_value = "test-api-key"
        
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
            mock_fetch.side_effect = Exception("HTTP Error")
            
            with pytest.raises(Exception):
                sportsbook_api.get_odds("0012400001")

    @patch("nba_predictor.data.sportsbook_api._get_api_key")
    def test_get_player_props_returns_dict(self, mock_api_key, clear_cache):
        """Test get_player_props returns dict."""
        from nba_predictor.data import sportsbook_api
        mock_api_key.return_value = "test-api-key"
        
        with patch("nba_predictor.data.sportsbook_api.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200, json=lambda: {"player_props": [{"player_id": "203999", "prop_type": "points", "odds": {"over": -110, "under": -110}}]})
            result = sportsbook_api.get_player_props("0012400001")
            
        assert isinstance(result, dict)
        assert "player_props" in result

    @patch("nba_predictor.data.sportsbook_api._get_api_key")
    def test_get_player_props_uses_cache(self, mock_api_key, clear_cache):
        """Test get_player_props uses cache."""
        from nba_predictor.data import sportsbook_api
        mock_api_key.return_value = "test-api-key"
        
        cache_path = clear_cache / "player_props_0012400001.json"
        cache_data = {"data": {"player_props": []}, "cached_at": 9999999999}
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
        
        with patch("nba_predictor.data.sportsbook_api.requests.get") as mock_get:
            result = sportsbook_api.get_player_props("0012400001")
            
        assert isinstance(result, dict)
        assert mock_get.called is False

    def test_complete_flow_with_realistic_data(self, clear_cache):
        """Test complete flow with realistic data."""
        from nba_predictor.data import sportsbook_api
        
        # Test cache_key function
        key = sportsbook_api.cache_key("0012400001")
        assert isinstance(key, str)
        
        # Test get_odds function exists and works
        with patch("nba_predictor.data.sportsbook_api._get_api_key") as mock_key:
            mock_key.return_value = "test-key"
            
            with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
                mock_fetch.return_value = {
                    "id": "0012400001",
                    "teams": ["Boston Celtics", "Brooklyn Nets"],
                    "player_props": [
                        {"player_id": "203999", "prop_type": "points", "odds": {"over": -110, "under": -110}},
                        {"player_id": "203540", "prop_type": "assists", "odds": {"over": -105, "under": -115}},
                        {"player_id": "1629630", "prop_type": "rebounds", "odds": {"over": -120, "under": +100}},
                    ],
                }
                result = sportsbook_api.get_odds("0012400001")
                
            assert isinstance(result, dict)
            
            # Get player props - uses requests.get directly, so patch that
            with patch("nba_predictor.data.sportsbook_api.requests.get") as mock_get:
                mock_get.return_value = MagicMock(status_code=200, json=lambda: {
                    "player_props": [
                        {"player_id": "203999", "prop_type": "points", "odds": {"over": -110, "under": -110}},
                        {"player_id": "203540", "prop_type": "assists", "odds": {"over": -105, "under": -115}},
                        {"player_id": "1629630", "prop_type": "rebounds", "odds": {"over": -120, "under": +100}},
                    ]
                })
                props = sportsbook_api.get_player_props("0012400001")
                
            assert isinstance(props, dict)
            assert "player_props" in props