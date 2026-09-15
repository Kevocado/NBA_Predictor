"""Tests for the odds_api module - The Odds API integration."""
import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Test constants
MOCK_BULK_ODDS_RESPONSE = [
    {
        "id": "a46a6a6a6a6a6a6a6a6a6a6a6a6a6a6a",
        "sport_key": "basketball_nba",
        "teams": ["Boston Celtics", "Brooklyn Nets"],
        "home_team": "Boston Celtics",
        "away_team": "Brooklyn Nets",
        "commence_time": "2024-01-15T00:00:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {"key": "h2h", "outcomes": [{"name": "Boston Celtics", "price": 1.85}, {"name": "Brooklyn Nets", "price": 1.95}]},
                    {"key": "spreads", "outcomes": [{"name": "Boston Celtics", "price": -1.5, "point": -5.5}, {"name": "Brooklyn Nets", "price": +1.5, "point": +5.5}]},
                    {"key": "totals", "outcomes": [{"name": "Over", "price": 1.90, "point": 225.5}, {"name": "Under", "price": 1.90, "point": 225.5}]}
                ]
            }
        ]
    }
]


class TestOddsApi:
    """Test suite for odds_api module."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test fixtures."""
        # Create cache directory
        self.cache_dir = tmp_path / "cache" / "odds"
        self.cache_dir.mkdir(parents=True)
        
        # Store original env variables
        self.original_api_key = os.environ.get('ODDS_API_KEY')
        os.environ['ODDS_API_KEY'] = "test_api_key"
        
        yield
        
        # Restore original env
        if self.original_api_key:
            os.environ['ODDS_API_KEY'] = self.original_api_key
        else:
            os.environ.pop('ODDS_API_KEY', None)

    def test_get_odds_success(self):
        """Test successful bulk fetch of odds data."""
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = MOCK_BULK_ODDS_RESPONSE
            mock_get.return_value = mock_response
            
            from nba_predictor.data import odds_api
            
            result = odds_api.get_odds()
            
            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]['id'] == "a46a6a6a6a6a6a6a6a6a6a6a6a6a6a6a"
            assert result[0]['sport_key'] == "basketball_nba"
            
            # Verify API was called with correct parameters
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert "basketball_nba" in call_args[0][0]
            assert "h2h,spreads,totals" in call_args[1]['params']['markets']
            assert "test_api_key" in call_args[1]['params']['apiKey']

    def test_get_odds_cached(self):
        """Test that odds are cached and reused."""
        # First, populate cache
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = MOCK_BULK_ODDS_RESPONSE
            mock_get.return_value = mock_response
            
            from nba_predictor.data import odds_api
            
            # First call - fetches from API
            result1 = odds_api.get_odds()
            
            # Second call - should use cache (mock should not be called again)
            mock_get.reset_mock()
            result2 = odds_api.get_odds()
        
        # Verify API was NOT called again
        assert mock_get.call_count == 0
        assert result1 == result2
        
        # Verify cache file was created
        cache_files = list(self.cache_dir.glob("*.json"))
        assert len(cache_files) >= 1

    @patch('nba_predictor.data.odds_api.requests.get')
    def test_get_odds_retries_on_failure(self, mock_get):
        """Test retry logic with exponential backoff."""
        # Setup mock to fail first call then succeed
        mock_get.side_effect = [
            MagicMock(status_code=503),
            MagicMock(status_code=200, json=lambda: MOCK_BULK_ODDS_RESPONSE)
        ]
        
        from nba_predictor.data import odds_api
        
        result = odds_api.get_odds()
        
        assert result == MOCK_BULK_ODDS_RESPONSE
        assert mock_get.call_count == 2

    @patch('nba_predictor.data.odds_api.requests.get')
    def test_get_odds_max_retries_exceeded(self, mock_get):
        """Test that max retries are respected."""
        # Setup mock to always fail
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        mock_get.side_effect = [mock_response] * 6  # 5 retries + 1 initial
        
        from nba_predictor.data import odds_api
        from tenacity import RetryError
        
        with pytest.raises(RetryError):
            odds_api.get_odds()
        
        assert mock_get.call_count == 6

    def test_get_h2h_odds_success(self):
        """Test successful fetch of head-to-head odds for a game."""
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = MOCK_BULK_ODDS_RESPONSE
            mock_get.return_value = mock_response
            
            from nba_predictor.data import odds_api
            
            # First get odds to populate cache
            odds_api.get_odds()
            
            # Then get h2h odds for specific game
            result = odds_api.get_h2h_odds("a46a6a6a6a6a6a6a6a6a6a6a6a6a6a6a")
            
            assert isinstance(result, dict)
            assert 'h2h' in result
            assert len(result['h2h']['outcomes']) == 2

    def test_get_spreads_odds_success(self):
        """Test successful fetch of spread odds for a game."""
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = MOCK_BULK_ODDS_RESPONSE
            mock_get.return_value = mock_response
            
            from nba_predictor.data import odds_api
            
            odds_api.get_odds()
            result = odds_api.get_spreads_odds("a46a6a6a6a6a6a6a6a6a6a6a6a6a6a6a")
            
            assert isinstance(result, dict)
            assert 'spreads' in result

    def test_get_totals_odds_success(self):
        """Test successful fetch of total odds for a game."""
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = MOCK_BULK_ODDS_RESPONSE
            mock_get.return_value = mock_response
            
            from nba_predictor.data import odds_api
            
            odds_api.get_odds()
            result = odds_api.get_totals_odds("a46a6a6a6a6a6a6a6a6a6a6a6a6a6a6a")
            
            assert isinstance(result, dict)
            assert 'totals' in result

    def test_get_h2h_odds_no_data(self):
        """Test handling when no odds data is returned."""
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = []
            mock_get.return_value = mock_response
            
            from nba_predictor.data import odds_api
            
            # This should return empty dict since game not found
            result = odds_api.get_h2h_odds("nonexistent_game_id")
            
            assert result == {}

    def test_get_odds_no_api_key(self):
        """Test that missing API key raises error."""
        # Temporarily unset API key
        original_key = os.environ.get('ODDS_API_KEY')
        os.environ.pop('ODDS_API_KEY', None)
        
        try:
            # Need to reload to pick up new env state
            from nba_predictor.data import odds_api
            import importlib
            importlib.reload(odds_api)
            
            with pytest.raises(ValueError, match="ODDS_API_KEY"):
                odds_api.get_odds()
        finally:
            # Restore API key
            if original_key:
                os.environ['ODDS_API_KEY'] = original_key
            else:
                os.environ.pop('ODDS_API_KEY', None)

    def test_get_odds_rate_limit_handling(self):
        """Test rate limit handling with 429 response."""
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            # First call returns 429, second succeeds
            mock_get.side_effect = [
                MagicMock(status_code=429),
                MagicMock(status_code=200, json=lambda: MOCK_BULK_ODDS_RESPONSE)
            ]
            
            from nba_predictor.data import odds_api
            
            result = odds_api.get_odds()
            
            assert result == MOCK_BULK_ODDS_RESPONSE
            assert mock_get.call_count == 2

    def test_bulk_odds_structure(self):
        """Test the structure of bulk odds response."""
        with patch('nba_predictor.data.odds_api.requests.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = MOCK_BULK_ODDS_RESPONSE
            mock_get.return_value = mock_response
            
            from nba_predictor.data import odds_api
            
            result = odds_api.get_odds()
            
            # Verify response structure
            assert isinstance(result, list)
            game = result[0]
            assert 'id' in game
            assert 'sport_key' in game
            assert 'teams' in game
            assert 'bookmakers' in game

    def test_cache_functions(self, tmp_path):
        """Test cache management functions."""
        from nba_predictor.data import odds_api
        
        # Test cache dir creation
        assert odds_api.CACHE_DIR.exists()
        
        # Test get_cache_info
        cache_info = odds_api.get_cache_info()
        assert 'file_count' in cache_info
        assert 'total_size_bytes' in cache_info
        assert 'cache_dir' in cache_info

    def test_clear_cache(self, tmp_path):
        """Test cache clearing."""
        from nba_predictor.data import odds_api
        
        # Create a test cache file
        test_file = odds_api.CACHE_DIR / "test_clear.json"
        test_file.write_text('{"test": "data"}')
        
        # Clear cache
        odds_api.clear_cache()
        
        # Verify file was deleted
        assert not test_file.exists()
