# tests/test_data_sportsbook_api.py
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import patch, MagicMock

# Mock the config before importing sportsbook_api
@pytest.fixture(autouse=True)
def mock_config(tmp_path):
    """Mock config module before importing sportsbook_api"""
    # Create a unique cache dir for each test run
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with patch.dict(os.environ, {"SPORTSBOOK_API_KEY": "test_api_key"}):
        # Create a mock module for config
        import types
        config_module = types.ModuleType('nba_predictor.config')
        config_module.SPORTSBOOK_API_KEY = "test_api_key"
        config_module.CACHE_DIR = cache_dir
        config_module.CACHE_SUBDIRS = ["nba_api", "balldontlie", "odds", "sportsbook", "injuries", "espn"]
        
        def ensure_cache_dirs():
            for subdir in config_module.CACHE_SUBDIRS:
                (config_module.CACHE_DIR / subdir).mkdir(parents=True, exist_ok=True)
        
        config_module.ensure_cache_dirs = ensure_cache_dirs
        
        # Inject into sys.modules
        import sys
        sys.modules['nba_predictor.config'] = config_module
        yield config_module


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing"""
    cache_dir = tmp_path / "cache" / "sportsbook"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class TestCacheKey:
    """Test cache_key function"""
    
    def test_cache_key_returns_string(self):
        from nba_predictor.data import sportsbook_api
        
        key = sportsbook_api.cache_key("0021900001")
        assert isinstance(key, str)
    
    def test_cache_key_format(self):
        from nba_predictor.data import sportsbook_api
        
        key = sportsbook_api.cache_key("0021900001")
        assert key == "odds_0021900001.json"
    
    def test_cache_key_unique_per_game_id(self):
        from nba_predictor.data import sportsbook_api
        
        key1 = sportsbook_api.cache_key("0021900001")
        key2 = sportsbook_api.cache_key("0021900002")
        assert key1 != key2


class TestGetOdds:
    """Test get_odds function"""
    
    def test_get_odds_returns_dict(self, temp_cache_dir, mock_config):
        from nba_predictor.data import sportsbook_api
        from datetime import timezone
        
        # Mock the API response
        mock_response = {
            "game_id": "0021900001",
            "odds": {
                "moneyline": {
                    "home": -150,
                    "away": +130
                },
                "spread": {
                    "home": -5.5,
                    "away": +5.5
                },
                "total": 225.5
            },
            "last_updated": "2023-01-15T10:30:00Z"
        }
        
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
            mock_fetch.return_value = mock_response
            
            result = sportsbook_api.get_odds("0021900001")
            
            assert isinstance(result, dict)
            assert result["game_id"] == "0021900001"
            assert "cached_at" in result  # Should have been added by _save_cache
    
    def test_get_odds_uses_cache_when_fresh(self, temp_cache_dir, mock_config):
        from nba_predictor.data import sportsbook_api
        from datetime import datetime, timezone
        
        game_id = "0021900001"
        cache_file = temp_cache_dir / f"odds_{game_id}.json"
        
        # Create a fresh cache file
        now = datetime.now(timezone.utc)
        cache_data = {
            "game_id": game_id,
            "odds": {"moneyline": {"home": -150, "away": 130}},
            "last_updated": now.isoformat(),
            "cached_at": now.timestamp()
        }
        cache_file.write_text(json.dumps(cache_data))
        
        # Should return cached data without calling API
        result = sportsbook_api.get_odds(game_id)
        assert result["game_id"] == game_id
    
    def test_get_odds_refreshes_expired_cache(self, temp_cache_dir, mock_config):
        from nba_predictor.data import sportsbook_api
        from datetime import datetime, timezone, timedelta
        
        game_id = "0021900001"
        cache_file = temp_cache_dir / f"odds_{game_id}.json"
        
        # Create an expired cache file (older than 6 hours)
        seven_hours_ago = datetime.now(timezone.utc) - timedelta(hours=7)
        cache_data = {
            "game_id": game_id,
            "odds": {"moneyline": {"home": -150, "away": 130}},
            "last_updated": seven_hours_ago.isoformat(),
            "cached_at": seven_hours_ago.timestamp()
        }
        cache_file.write_text(json.dumps(cache_data))
        
        # Mock fresh API response
        now = datetime.now(timezone.utc)
        fresh_response = {
            "game_id": game_id,
            "odds": {"moneyline": {"home": -145, "away": 125}},
            "last_updated": now.isoformat(),
        }
        
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
            mock_fetch.return_value = fresh_response
            
            result = sportsbook_api.get_odds(game_id)
            
            # Should fetch new data and update cache
            assert mock_fetch.called is True, "API fetch should have been called for expired cache"
            assert result["game_id"] == game_id
            # Verify the API was called with correct game_id
            mock_fetch.assert_called_once_with(game_id)
    
    def test_get_odds_handles_api_error(self, temp_cache_dir, mock_config):
        import requests
        
        from nba_predictor.data import sportsbook_api
        from tenacity import RetryError
        
        # Mock API to raise an exception
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_fetch:
            mock_fetch.side_effect = requests.exceptions.HTTPError("HTTP Error")
            
            # Should retry and eventually raise RetryError after max retries
            with pytest.raises(RetryError):
                sportsbook_api.get_odds("0021900001")


class TestGetPlayerProps:
    """Test get_player_props function"""
    
    def test_get_player_props_returns_dict(self, temp_cache_dir, mock_config):
        from nba_predictor.data import sportsbook_api
        from datetime import timezone
        
        mock_response = {
            "game_id": "0021900001",
            "player_props": [
                {
                    "player_id": "203999",
                    "player_name": "LeBron James",
                    "prop_type": "points",
                    "odds": {
                        "over": -110,
                        "under": -110,
                        "line": 27.5
                    }
                }
            ],
            "last_updated": "2023-01-15T10:30:00Z"
        }
        
        with patch("nba_predictor.data.sportsbook_api._fetch_player_props_api") as mock_fetch:
            mock_fetch.return_value = mock_response
            
            result = sportsbook_api.get_player_props("0021900001")
            
            assert isinstance(result, dict)
            assert result["game_id"] == "0021900001"
            assert len(result["player_props"]) > 0
            assert "cached_at" in result  # Should have been added by _save_cache
    
    def test_get_player_props_uses_cache(self, temp_cache_dir, mock_config):
        from nba_predictor.data import sportsbook_api
        from datetime import datetime, timezone
        
        game_id = "0021900001"
        cache_file = temp_cache_dir / f"player_props_{game_id}.json"
        
        now = datetime.now(timezone.utc)
        cache_data = {
            "game_id": game_id,
            "player_props": [],
            "cached_at": now.timestamp()
        }
        cache_file.write_text(json.dumps(cache_data))
        
        result = sportsbook_api.get_player_props(game_id)
        assert isinstance(result, dict)


class TestIntegration:
    """Integration tests with real API structure (mocked)"""
    
    def test_complete_flow_with_realistic_data(self, temp_cache_dir, mock_config):
        from nba_predictor.data import sportsbook_api
        from datetime import timezone
        
        # Test the complete flow with realistic NBA data structure
        game_id = "0021900001"
        
        odds_response = {
            "game_id": game_id,
            "odds": {
                "moneyline": {
                    "home": -150,
                    "away": +130
                },
                "spread": {
                    "home": -5.5,
                    "away": +5.5
                },
                "total": 225.5
            },
            "last_updated": "2023-01-15T10:30:00Z"
        }
        
        props_response = {
            "game_id": game_id,
            "player_props": [
                {
                    "player_id": "203999",
                    "player_name": "LeBron James",
                    "team": "LAL",
                    "prop_type": "points",
                    "odds": {
                        "over": -110,
                        "under": -110,
                        "line": 27.5
                    }
                },
                {
                    "player_id": "203999",
                    "player_name": "LeBron James",
                    "team": "LAL",
                    "prop_type": "assists",
                    "odds": {
                        "over": -105,
                        "under": -115,
                        "line": 8.5
                    }
                },
                {
                    "player_id": "201933",
                    "player_name": "Stephen Curry",
                    "team": "GSW",
                    "prop_type": "points",
                    "odds": {
                        "over": -115,
                        "under": -105,
                        "line": 25.5
                    }
                }
            ],
            "last_updated": "2023-01-15T10:30:00Z"
        }
        
        with patch("nba_predictor.data.sportsbook_api._fetch_odds_api") as mock_odds:
            with patch("nba_predictor.data.sportsbook_api._fetch_player_props_api") as mock_props:
                mock_odds.return_value = odds_response
                mock_props.return_value = props_response
                
                odds = sportsbook_api.get_odds(game_id)
                
                # Get player_props from a new game ID to avoid cache conflict
                props_game_id = "0021900002"
                props = sportsbook_api.get_player_props(props_game_id)
                
                # Verify odds data
                assert "odds" in odds
                assert "moneyline" in odds["odds"]
                assert "spread" in odds["odds"]
                assert "total" in odds["odds"]
                
                # Verify player props data
                assert "player_props" in props
                assert len(props["player_props"]) == 3
                
                # Verify player names
                player_names = [p["player_name"] for p in props["player_props"]]
                assert "LeBron James" in player_names
                assert "Stephen Curry" in player_names
