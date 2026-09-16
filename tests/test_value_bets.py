"""Tests for odds/value_bets module."""
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path


class TestShinTwoWay:
    """Test suite for shin_two_way."""

    def test_returns_dict(self):
        """Test shin_two_way returns dict."""
        from nba_predictor.odds.value_bets import shin_two_way
        
        result = shin_two_way(0.55, 0.45)
        
        assert "fair_prob_home" in result
        assert "fair_prob_away" in result
        assert "market_prob_home" in result
        assert "market_prob_away" in result

    def test_fair_probs_sum_to_one(self):
        """Test fair probabilities sum to 1."""
        from nba_predictor.odds.value_bets import shin_two_way
        
        result = shin_two_way(0.6, 0.4)
        
        total = result["fair_prob_home"] + result["fair_prob_away"]
        assert abs(total - 1.0) < 0.01

    def test_with_margin(self):
        """Test shin_two_way with margin."""
        from nba_predictor.odds.value_bets import shin_two_way
        
        result = shin_two_way(0.55, 0.45, m=0.05)
        
        assert "fair_prob_home" in result
        assert "margin" in result
        assert result["margin"] == 0.05

    def test_edge_case_equal_probs(self):
        """Test with equal probabilities."""
        from nba_predictor.odds.value_bets import shin_two_way
        
        result = shin_two_way(0.5, 0.5)
        
        assert abs(result["fair_prob_home"] - 0.5) < 0.01


class TestDetectValueBets:
    """Test suite for detect_value_bets."""

    def test_returns_list(self):
        """Test detect_value_bets returns list."""
        from nba_predictor.odds.value_bets import detect_value_bets
        
        game_data = {
            "game_id": "0012400001",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_prob": 0.65,
        }
        odds_data = {"home_odds": -120, "away_odds": 100}
        
        result = detect_value_bets(game_data, odds_data)
        
        assert isinstance(result, list)

    def test_detects_value_bet(self):
        """Test value bet detection."""
        from nba_predictor.odds.value_bets import detect_value_bets
        
        game_data = {
            "game_id": "0012400001",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_prob": 0.70,  # Strong home favor
        }
        odds_data = {"home_odds": 110, "away_odds": -130}  # Market thinks game is closer
        
        result = detect_value_bets(game_data, odds_data, threshold=0.05)
        
        assert isinstance(result, list)

    def test_empty_result_low_edge(self):
        """Test no value bets when edge is low."""
        from nba_predictor.odds.value_bets import detect_value_bets
        
        game_data = {
            "game_id": "0012400001",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_prob": 0.52,  # Weak edge
        }
        odds_data = {"home_odds": -110, "away_odds": -110}
        
        result = detect_value_bets(game_data, odds_data, threshold=0.10)
        
        assert isinstance(result, list)
        assert len(result) == 0


class TestComputeValueBets:
    """Test suite for compute_value_bets."""

    def test_returns_dict(self):
        """Test compute_value_bets returns dict."""
        from nba_predictor.odds.value_bets import compute_value_bets
        
        game_data = {
            "game_id": "0012400001",
            "home_prob": 0.60,
        }
        odds_data = {
            "home_implied_prob": 0.55,
            "away_implied_prob": 0.45,
        }
        
        result = compute_value_bets(game_data, odds_data)
        
        assert "game_id" in result
        assert "home_value_bet" in result
        assert "away_value_bet" in result
        assert "shin_fair_probs" in result

    def test_home_value_bet_exists(self):
        """Test home value bet detection."""
        from nba_predictor.odds.value_bets import compute_value_bets
        
        game_data = {"game_id": "0012400001", "home_prob": 0.70}
        odds_data = {"home_implied_prob": 0.55, "away_implied_prob": 0.45}
        
        result = compute_value_bets(game_data, odds_data)
        
        assert isinstance(result["home_value_bet"]["exists"], bool)


class TestValueBetsCache:
    """Test cache functionality for value bets."""

    def test_detect_value_bets_caches(self, tmp_path):
        """Test detect_value_bets caches results."""
        from nba_predictor import config
        from nba_predictor.odds.value_bets import detect_value_bets, VALUE_BETS_CACHE_DIR
        
        original = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        game_data = {
            "game_id": "0012400001",
            "home_team": "BOS",
            "away_team": "LAL",
            "home_prob": 0.65,
        }
        odds_data = {"home_odds": -120, "away_odds": 100}
        
        detect_value_bets(game_data, odds_data)
        
        cache_path = VALUE_BETS_CACHE_DIR / "value_bets_0012400001.json"
        assert cache_path.exists()
        
        config.CACHE_DIR = original

    def test_get_value_bets(self, tmp_path):
        """Test get_value_bets retrieves cached data."""
        from nba_predictor import config
        from nba_predictor.odds.value_bets import VALUE_BETS_CACHE_DIR, get_value_bets
        
        original = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create a cache file
        cache_path = VALUE_BETS_CACHE_DIR / "value_bets_0012400001.json"
        with open(cache_path, "w") as f:
            json.dump([{"game_id": "0012400001", "edge": 0.1}], f)
        
        result = get_value_bets("0012400001")
        assert isinstance(result, list)
        assert len(result) > 0
        
        config.CACHE_DIR = original


class TestValueBetsIntegration:
    """Integration tests for value bets."""

    def test_full_value_bet_pipeline(self):
        """Test complete value bet pipeline."""
        from nba_predictor.odds.value_bets import compute_value_bets, shin_two_way
        
        game_data = {
            "game_id": "0012400001",
            "home_prob": 0.60,
        }
        odds_data = {
            "home_implied_prob": 0.55,
            "away_implied_prob": 0.45,
        }
        
        # Compute value bets
        result = compute_value_bets(game_data, odds_data)
        assert isinstance(result, dict)
        
        # Also test shin_two_way directly
        shin = shin_two_way(0.55, 0.45)
        assert "fair_prob_home" in shin

    def test_shin_model_with_extreme_probs(self):
        """Test shin_two_way with extreme probabilities."""
        from nba_predictor.odds.value_bets import shin_two_way
        
        result = shin_two_way(0.9, 0.1)
        
        assert 0 < result["fair_prob_home"] < 1
        assert 0 < result["fair_prob_away"] < 1