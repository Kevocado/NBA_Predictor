"""Tests for models/__init__.py."""
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path
import numpy as np


class TestGameOutcomeModel:
    """Test suite for GameOutcomeModel."""

    def test_predict_returns_dict(self):
        """Test predict returns dict."""
        from nba_predictor.models import GameOutcomeModel
        
        model = GameOutcomeModel()
        game_data = {
            "home": {"net_rating": 5.0, "effective_field_goal_percentage": 0.55},
            "away": {"net_rating": 2.0, "effective_field_goal_percentage": 0.50},
            "home_court": True,
            "game_id": "0012400001",
        }
        
        result = model.predict(game_data)
        
        assert "prediction" in result
        assert "probability" in result
        assert "confidence" in result
        assert result["prediction"] in ["home", "away"]

    def test_predict_home_prob(self):
        """Test predict returns home probability."""
        from nba_predictor.models import GameOutcomeModel
        
        model = GameOutcomeModel()
        game_data = {
            "home": {"net_rating": 10.0},
            "away": {"net_rating": -5.0},
            "home_court": True,
        }
        
        result = model.predict(game_data)
        
        assert result["home_prob"] > 0.5
        assert result["prediction"] == "home"

    def test_fit(self):
        """Test model training."""
        from nba_predictor.models import GameOutcomeModel
        
        model = GameOutcomeModel()
        X = np.array([[1.0, 0.1, 1.0, 0.5, 1.0], [-1.0, -0.1, 0.0, -0.5, 0.0]])
        y = np.array([1.0, 0.0])
        
        model.fit(X, y, epochs=50, lr=0.1)
        
        assert model.trained is True
        assert model.weights is not None

    def test_get_model_info(self):
        """Test get_model_info returns dict."""
        from nba_predictor.models import GameOutcomeModel
        
        model = GameOutcomeModel()
        result = model.get_model_info()
        
        assert result["model_type"] == "GameOutcomeModel"
        assert "trained" in result

    def test_cache_result(self, tmp_path):
        """Test predict caches results."""
        from nba_predictor import config
        from nba_predictor.models import GameOutcomeModel, MODEL_CACHE_DIR
        
        original = config.CACHE_DIR
        config.CACHE_DIR = tmp_path / "cache"
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        # Update MODEL_CACHE_DIR to use the new cache dir
        import nba_predictor.models as models_module
        original_model_dir = models_module.MODEL_CACHE_DIR
        models_module.MODEL_CACHE_DIR = config.CACHE_DIR / "models"
        models_module.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        model = GameOutcomeModel()
        game_data = {
            "home": {"net_rating": 5.0},
            "away": {"net_rating": 2.0},
            "home_court": True,
            "game_id": "0012400001",
        }
        
        result = model.predict(game_data)
        
        cache_path = models_module.MODEL_CACHE_DIR / "game_outcome_0012400001.json"
        assert cache_path.exists()
        
        config.CACHE_DIR = original
        models_module.MODEL_CACHE_DIR = original_model_dir


class TestSpreadModel:
    """Test suite for SpreadModel."""

    def test_predict_returns_dict(self):
        """Test predict returns dict."""
        from nba_predictor.models import SpreadModel
        
        model = SpreadModel()
        game_data = {
            "home_rating": 1505,
            "away_rating": 1498,
            "game_id": "0012400001",
        }
        
        result = model.predict(game_data)
        
        assert "predicted_margin" in result
        assert "spread" in result
        assert "home_cover_probability" in result

    def test_predict_with_spread(self):
        """Test predict with spread data."""
        from nba_predictor.models import SpreadModel
        
        model = SpreadModel()
        game_data = {
            "home_rating": 1510,
            "away_rating": 1490,
            "spread": 5.0,
            "game_id": "0012400001",
        }
        
        result = model.predict(game_data)
        
        assert "spread_result" in result

    def test_get_model_info(self):
        """Test get_model_info returns dict."""
        from nba_predictor.models import SpreadModel
        
        model = SpreadModel()
        result = model.get_model_info()
        
        assert result["model_type"] == "SpreadModel"


class TestPlayerPropsModel:
    """Test suite for PlayerPropsModel."""

    def test_predict_returns_dict(self):
        """Test predict returns dict."""
        from nba_predictor.models import PlayerPropsModel
        
        model = PlayerPropsModel()
        player_data = {
            "player_id": "203999",
            "avg_points": 25.0,
            "avg_rebounds": 7.0,
            "avg_assists": 5.0,
            "minutes": 32.0,
            "opponent_defensive_rating": 110.0,
            "pace": 100.0,
        }
        
        result = model.predict(player_data)
        
        assert "points" in result
        assert "rebounds" in result
        assert "assists" in result
        assert "over_prob" in result["points"]

    def test_get_model_info(self):
        """Test get_model_info returns dict."""
        from nba_predictor.models import PlayerPropsModel
        
        model = PlayerPropsModel()
        result = model.get_model_info()
        
        assert result["model_type"] == "PlayerPropsModel"


class TestManifestModel:
    """Test suite for ManifestModel."""

    def test_generate_manifest(self):
        """Test generate_manifest returns dict."""
        from nba_predictor.models import ManifestModel
        
        model = ManifestModel()
        game_data = {
            "game_id": "0012400001",
            "date": "2024-10-25",
            "home_team": {"abbreviation": "BOS"},
            "away_team": {"abbreviation": "LAL"},
            "season": "2024-25",
        }
        
        result = model.generate_manifest(game_data)
        
        assert result["game_id"] == "0012400001"
        assert result["home_team"]["abbreviation"] == "BOS"
        assert result["season"] == "2024-25"

    def test_get_model_info(self):
        """Test get_model_info returns dict."""
        from nba_predictor.models import ManifestModel
        
        model = ManifestModel()
        result = model.get_model_info()
        
        assert result["model_type"] == "ManifestModel"


class TestModelsIntegration:
    """Integration tests for models."""

    def test_game_outcome_model_with_training(self):
        """Test GameOutcomeModel after training."""
        from nba_predictor.models import GameOutcomeModel
        
        model = GameOutcomeModel()
        X = np.array([[1.0, 0.2, 1.0, 0.5, 1.0], [-1.0, -0.2, 0.0, -0.5, 0.0]])
        y = np.array([1.0, 0.0])
        model.fit(X, y, epochs=100, lr=0.05)
        
        game_data = {
            "home": {"net_rating": 5.0, "effective_field_goal_percentage": 0.55},
            "away": {"net_rating": 2.0},
            "home_court": True,
        }
        
        result = model.predict(game_data)
        assert isinstance(result["probability"], float)
        assert result["probability"] > 0.0
        assert result["probability"] < 1.0

    def test_spread_model_with_training(self):
        """Test SpreadModel works correctly."""
        from nba_predictor.models import SpreadModel
        
        model = SpreadModel()
        game_data = {
            "home_rating": 1505,
            "away_rating": 1490,
            "game_id": "0012400001",
        }
        
        result = model.predict(game_data)
        assert isinstance(result["predicted_margin"], float)