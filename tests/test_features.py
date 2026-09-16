"""Tests for features/build module."""
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path


class TestComputeFourFactors:
    """Test suite for compute_four_factors."""

    def test_returns_dict_with_home_and_away(self):
        """Test compute_four_factors returns dict."""
        from nba_predictor.features.build import compute_four_factors
        
        game_data = {
            "home_stats": {"efg": 0.55, "tov_pct": 0.12, "orb_pct": 0.30, "ft_rate": 0.25},
            "away_stats": {"efg": 0.50, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.20},
        }
        
        result = compute_four_factors(game_data)
        
        assert "home" in result
        assert "away" in result
        assert "effective_field_goal_percentage" in result["home"]

    def test_default_values(self):
        """Test compute_four_factors with missing data."""
        from nba_predictor.features.build import compute_four_factors
        
        result = compute_four_factors({})
        
        assert "home" in result
        assert "away" in result


class TestComputeEfficiency:
    """Test suite for compute_efficiency."""

    def test_returns_dict(self):
        """Test compute_efficiency returns dict."""
        from nba_predictor.features.build import compute_efficiency
        
        game_data = {
            "home_stats": {"points": 110, "possessions": 100, "opponent_points": 100},
            "away_stats": {"points": 100, "possessions": 100, "opponent_points": 110},
        }
        
        result = compute_efficiency(game_data)
        
        assert "home" in result
        assert "net_rating" in result["home"]


class TestComputePowerRating:
    """Test suite for compute_power_rating."""

    def test_returns_dict(self):
        """Test compute_power_rating returns dict."""
        from nba_predictor.features.build import compute_power_rating
        
        result = compute_power_rating("BOS", "LAL", [])
        
        assert "home_power_rating" in result
        assert "away_power_rating" in result
        assert "home_court_advantage" in result


class TestComputeRestFatigue:
    """Test suite for compute_rest_fatigue."""

    def test_returns_dict(self):
        """Test compute_rest_fatigue returns dict."""
        from nba_predictor.features.build import compute_rest_fatigue
        
        result = compute_rest_fatigue(1, [])
        
        assert "rest_days" in result
        assert "back_to_back" in result
        assert "congestion_3_in_4" in result


class TestComputeTravel:
    """Test suite for compute_travel."""

    def test_returns_dict(self):
        """Test compute_travel returns dict."""
        from nba_predictor.features.build import compute_travel
        
        result = compute_travel(1, [])
        
        assert "total_mileage" in result
        assert "fatigue_index" in result


class TestComputeInjuryImpact:
    """Test suite for compute_injury_impact."""

    def test_returns_dict(self):
        """Test compute_injury_impact returns dict."""
        from nba_predictor.features.build import compute_injury_impact
        
        game_data = {"injuries": [{"status": "Out", "player_name": "Test"}]}
        
        result = compute_injury_impact(game_data)
        
        assert "missing_player_value" in result
        assert "missing_players" in result


class TestComputeContext:
    """Test suite for compute_context."""

    def test_returns_dict(self):
        """Test compute_context returns dict."""
        from nba_predictor.features.build import compute_context
        
        game_data = {
            "home_team": {"abbreviation": "BOS"},
            "away_team": {"abbreviation": "LAL"},
        }
        
        result = compute_context(game_data)
        
        assert "altitude_flag" in result
        assert "conference_game" in result


class TestBuildFeatures:
    """Test suite for build_features."""

    def test_returns_dict(self):
        """Test build_features returns dict."""
        from nba_predictor.features.build import build_features
        
        game_data = {
            "game_id": "0012400001",
            "home_team": {"abbreviation": "BOS"},
            "away_team": {"abbreviation": "LAL"},
            "home_stats": {"efg": 0.55, "tov_pct": 0.12, "orb_pct": 0.30, "ft_rate": 0.25},
            "away_stats": {"efg": 0.50, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.20},
        }
        team_data = {"recent_games": []}
        injury_data = {"injuries": []}
        
        result = build_features(game_data, team_data, injury_data)
        
        assert "game_id" in result
        assert "four_factors" in result
        assert "efficiency" in result
        assert "power_rating" in result

    def test_caches_features(self, tmp_path):
        """Test build_features caches results."""
        from nba_predictor import config
        from nba_predictor.features.build import build_features, FEATURES_CACHE_DIR
        
        original = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        game_data = {
            "game_id": "0012400001",
            "home_team": {"abbreviation": "BOS"},
            "away_team": {"abbreviation": "LAL"},
        }
        team_data = {"recent_games": []}
        injury_data = {"injuries": []}
        
        build_features(game_data, team_data, injury_data)
        
        cache_path = FEATURES_CACHE_DIR / "features_0012400001.json"
        assert cache_path.exists()
        
        config.CACHE_DIR = original

    def test_get_cached_features(self, tmp_path):
        """Test get_cached_features."""
        from nba_predictor import config
        from nba_predictor.features.build import FEATURES_CACHE_DIR, get_cached_features, build_features
        
        original = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        game_data = {"game_id": "0012400001"}
        team_data = {"recent_games": []}
        injury_data = {"injuries": []}
        build_features(game_data, team_data, injury_data)
        
        result = get_cached_features("0012400001")
        assert result is not None
        assert "game_id" in result
        
        config.CACHE_DIR = original


class TestFeaturesIntegration:
    """Integration tests for features."""

    def test_full_feature_vector(self):
        """Test complete feature vector generation."""
        from nba_predictor.features.build import build_features
        
        game_data = {
            "game_id": "0012400001",
            "home_team": {"abbreviation": "BOS", "id": 1},
            "away_team": {"abbreviation": "LAL", "id": 2},
            "home_stats": {"points": 110, "possessions": 100, "opponent_points": 100, "efg": 0.55, "tov_pct": 0.12, "orb_pct": 0.30, "ft_rate": 0.25},
            "away_stats": {"points": 100, "possessions": 100, "opponent_points": 110, "efg": 0.50, "tov_pct": 0.15, "orb_pct": 0.25, "ft_rate": 0.20},
        }
        team_data = {"recent_games": [{"location": {}}]}
        injury_data = {"injuries": []}
        
        features = build_features(game_data, team_data, injury_data)
        
        assert isinstance(features, dict)
        assert features["game_id"] == "0012400001"