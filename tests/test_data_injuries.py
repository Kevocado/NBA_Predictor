# tests/test_data_injuries.py
import json
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import patch, MagicMock

# Mock the config before importing injuries
@pytest.fixture(autouse=True)
def mock_config():
    """Mock config module before importing injuries"""
    with patch.dict(os.environ, {}, clear=False):
        # Create a mock module for config
        import types
        config_module = types.ModuleType('nba_predictor.config')
        config_module.BALLDONTLIE_API_KEY = None
        config_module.SPORTSBOOK_API_KEY = None
        config_module.ODDS_API_KEY = None
        config_module.CACHE_DIR = Path("/tmp/nba_cache_test")
        config_module.CACHE_SUBDIRS = ["nba_api", "balldontlie", "odds", "sportsbook", "injuries", "espn"]
        
        def ensure_cache_dirs():
            (config_module.CACHE_DIR / "injuries").mkdir(parents=True, exist_ok=True)
        
        config_module.ensure_cache_dirs = ensure_cache_dirs
        
        # Inject into sys.modules
        import sys
        sys.modules['nba_predictor.config'] = config_module
        yield config_module


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for testing"""
    cache_dir = tmp_path / "cache" / "injuries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


class TestGetCurrentInjuries:
    """Test get_current_injuries function"""
    
    def test_get_current_injuries_returns_list(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Mock the API response
        mock_response = [
            {
                "player_id": 203999,
                "player_name": "LeBron James",
                "team_id": 1610612747,
                "team_abbreviation": "LAL",
                "description": "Lower extremity injury",
                "status": "Questionable",
                "date": "2023-01-15",
                "detail": "Out with right knee soreness"
            }
        ]
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_injury_report") as mock_fetch:
                mock_fetch.return_value = mock_response
                
                result = injuries.get_current_injuries()
                
                assert isinstance(result, list)
                assert len(result) > 0
                assert result[0]["player_name"] == "LeBron James"
    
    def test_get_current_injuries_empty_report(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_injury_report") as mock_fetch:
                mock_fetch.return_value = []
                
                result = injuries.get_current_injuries()
                
                assert result == []
    
    def test_get_current_injuries_uses_cache(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        # Create a fresh cache file
        cache_data = {
            "injuries": [
                {
                    "player_id": 203999,
                    "player_name": "LeBron James",
                    "status": "Questionable"
                }
            ],
            "cached_at": datetime.now(timezone.utc).timestamp()
        }
        cache_file = temp_cache_dir / "injury_report.json"
        cache_file.write_text(json.dumps(cache_data))
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            # Should return cached data without calling API
            result = injuries.get_current_injuries()
            assert isinstance(result, list)


class TestGetPlayerInjuryHistory:
    """Test get_player_injury_history function"""
    
    def test_get_player_injury_history_returns_list(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        mock_response = [
            {
                "player_id": 203999,
                "player_name": "LeBron James",
                "team_id": 1610612747,
                "team_abbreviation": "LAL",
                "date": "2023-01-10",
                "status": "Out",
                "description": "Right knee soreness",
                "return_date": "2023-01-15"
            },
            {
                "player_id": 203999,
                "player_name": "LeBron James",
                "team_id": 1610612747,
                "team_abbreviation": "LAL",
                "date": "2022-12-20",
                "status": "Out",
                "description": "Calf strain",
                "return_date": "2023-01-03"
            }
        ]
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_player_injury_history") as mock_fetch:
                mock_fetch.return_value = mock_response
                
                result = injuries.get_player_injury_history(203999)
                
                assert isinstance(result, list)
                assert len(result) == 2
                assert result[0]["player_name"] == "LeBron James"
    
    def test_get_player_injury_history_no_history(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_player_injury_history") as mock_fetch:
                mock_fetch.return_value = []
                
                result = injuries.get_player_injury_history(999999)
                
                assert result == []
    
    def test_get_player_injury_history_uses_cache(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        # Create a fresh cache file
        cache_data = {
            "player_id": 203999,
            "injuries": [
                {
                    "date": "2023-01-10",
                    "status": "Out"
                }
            ],
            "cached_at": datetime.now(timezone.utc).timestamp()
        }
        cache_file = temp_cache_dir / "player_203999_history.json"
        cache_file.write_text(json.dumps(cache_data))
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            # Should return cached data without calling API
            result = injuries.get_player_injury_history(203999)
            assert isinstance(result, list)


class TestGetMissingPlayerValue:
    """Test get_missing_player_value function"""
    
    def test_get_missing_player_value_returns_float(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_player_stats") as mock_stats:
                mock_stats.return_value = {
                    "player_id": 203999,
                    "season": 2022,
                    "games_played": 60,
                    "pts": 25.7,
                    "reb": 7.9,
                    "ast": 8.3,
                    "usage_rate": 32.5,
                    "true_shooting_pct": 0.582
                }
                
                result = injuries.get_missing_player_value(203999, 2022)
                
                assert isinstance(result, float)
                assert result > 0
    
    def test_get_missing_player_value_no_stats_returns_zero(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_player_stats") as mock_stats:
                mock_stats.return_value = None
                
                result = injuries.get_missing_player_value(999999, 2022)
                
                assert result == 0.0
    
    def test_get_missing_player_value_uses_cache(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        # Create a fresh cache file with exact value
        cache_data = {
            "player_id": 203999,
            "season": 2022,
            "production_value": 28.5,
            "cached_at": datetime.now(timezone.utc).timestamp()
        }
        cache_file = temp_cache_dir / "player_203999_season_2022_value.json"
        cache_file.write_text(json.dumps(cache_data))
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            # Should return cached data without calling API
            result = injuries.get_missing_player_value(203999, 2022)
            assert isinstance(result, float)
            assert result == 28.5


class TestIntegration:
    """Integration tests with realistic data structures"""
    
    def test_complete_injury_report_flow(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        # Mock injury report data
        injury_report = [
            {
                "player_id": 203999,
                "player_name": "LeBron James",
                "team_id": 1610612747,
                "team_abbreviation": "LAL",
                "description": "Lower extremity injury",
                "status": "Questionable",
                "date": "2023-01-15",
                "detail": "Out with right knee soreness"
            },
            {
                "player_id": 201933,
                "player_name": "Stephen Curry",
                "team_id": 1610612744,
                "team_abbreviation": "GSW",
                "description": "Ankle sprain",
                "status": "Out",
                "date": "2023-01-14",
                "detail": "Out with ankle sprain"
            }
        ]
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_injury_report") as mock_fetch:
                mock_fetch.return_value = injury_report
                
                # Get current injuries
                current = injuries.get_current_injuries()
                assert len(current) == 2
                
                # Get injury history for one player
                history = injuries.get_player_injury_history(203999)
                assert isinstance(history, list)
    
    def test_missing_player_value_calculation(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        # Mock realistic player stats
        player_stats = {
            "player_id": 203999,
            "player_name": "LeBron James",
            "season": 2022,
            "games_played": 60,
            "minutes_per_game": 38.6,
            "pts": 25.7,
            "reb": 7.9,
            "ast": 8.3,
            "usage_rate": 32.5,
            "true_shooting_pct": 0.582,
            "win_shares": 14.2
        }
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_player_stats") as mock_stats:
                mock_stats.return_value = player_stats
                
                value = injuries.get_missing_player_value(203999, 2022)
                
                # Should compute a reasonable production value
                assert value > 0
                # Win shares per 48 minutes ≈ 14.2 / 60 * 48 ≈ 11.3
                # This should be reflected in the production value
                assert value < 100  # Production value should be reasonable


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_api_error_triggers_retry(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            with patch("nba_predictor.data.injuries._fetch_injury_report") as mock_fetch:
                mock_fetch.side_effect = Exception("API Error")
                
                # Should raise after max retries
                with pytest.raises(Exception, match="API Error"):
                    injuries.get_current_injuries()
    
    def test_cache_file_corrupted(self, temp_cache_dir, mock_config):
        from nba_predictor.data import injuries
        
        # Patch _get_cache_path to use temp directory
        def mock_get_cache_path(filename):
            return temp_cache_dir / filename
        
        # Create a corrupted cache file
        cache_file = temp_cache_dir / "injury_report.json"
        cache_file.write_text("not valid json")
        
        with patch("nba_predictor.data.injuries._get_cache_path", side_effect=mock_get_cache_path):
            # Should handle gracefully - when cache is corrupted, it should return sample data
            # The current implementation returns sample data when _fetch_injury_report returns
            with patch("nba_predictor.data.injuries._fetch_injury_report") as mock_fetch:
                mock_fetch.return_value = []
                
                result = injuries.get_current_injuries()
                assert result == []
