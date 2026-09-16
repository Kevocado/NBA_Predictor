"""Machine learning models for NBA game predictions.

Contains:
- GameOutcomeModel: Predicts win/loss outcome
- SpreadModel: Predicts point spread results
- PlayerPropsModel: Predicts player proposition stats
- ManifestModel: Generates game manifest data
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
from nba_predictor import config
from nba_predictor.features.build import build_features

# Cache directory for model predictions
MODEL_CACHE_DIR = config.CACHE_DIR / "models"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class GameOutcomeModel:
    """Predicts win/loss outcome for NBA games.
    
    Uses logistic regression on engineered features to predict
    the probability of a home team win.
    """
    
    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.trained = False
    
    def _sigmoid(self, z: float) -> float:
        """Sigmoid activation function."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def _extract_features(self, game_data: dict) -> np.ndarray:
        """Extract numerical features from game data."""
        home = game_data.get("home", {})
        away = game_data.get("away", {})
        
        home_rating = home.get("net_rating", 0)
        away_rating = away.get("net_rating", 0)
        home_four_factor = home.get("effective_field_goal_percentage", 0.5)
        away_four_factor = away.get("effective_field_goal_percentage", 0.5)
        home_rest = game_data.get("home_rest_days", 2)
        away_rest = game_data.get("away_rest_days", 2)
        home_travel = game_data.get("home_travel", 0)
        away_travel = game_data.get("away_travel", 0)
        
        return np.array([
            home_rating - away_rating,
            home_four_factor - away_four_factor,
            home_rest - away_rest,
            home_travel - away_travel,
            1.0 if home.get("home_court") else 0.0,
        ])
    
    def predict(self, game_data: dict) -> dict:
        """Predict game outcome.
        
        Args:
            game_data: Dictionary containing game information
            
        Returns:
            Dict with prediction, probability, and confidence
        """
        features = self._extract_features(game_data)
        
        if self.trained:
            z = np.dot(self.weights, features) + self.bias
            prob = self._sigmoid(z)
        else:
            # Fallback: use rating differential
            rating_diff = float(features[0])
            prob = self._sigmoid(rating_diff * 0.1)
        
        confidence = abs(2 * prob - 1)
        
        prediction = "home" if prob > 0.5 else "away"
        
        result = {
            "prediction": prediction,
            "probability": float(prob),
            "confidence": float(confidence),
            "home_prob": float(prob),
        }
        
        # Cache result
        game_id = game_data.get("game_id", "")
        cache_path = MODEL_CACHE_DIR / f"game_outcome_{game_id}.json"
        with open(cache_path, "w") as f:
            json.dump(result, f)
        
        return result
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 0.01) -> None:
        """Train the model using logistic regression.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0 or 1)
            epochs: Number of training epochs
            lr: Learning rate
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(epochs):
            z = X @ self.weights + self.bias
            predictions = self._sigmoid(z)
            
            error = predictions - y
            self.weights -= lr * X.T @ error / n_samples
            self.bias -= lr * np.mean(error)
        
        self.trained = True
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "GameOutcomeModel",
            "trained": self.trained,
            "num_features": len(self.weights) if self.weights is not None else 0,
        }


class SpreadModel:
    """Predicts point spread results for NBA games.
    
    Uses Elo ratings and feature engineering to predict
    the margin of victory and compare against the spread.
    """
    
    def __init__(self):
        self.home_court_advantage = 3.0
        self.trained = False
    
    def predict(self, game_data: dict) -> dict:
        """Predict spread result.
        
        Args:
            game_data: Dictionary containing game information
            
        Returns:
            Dict with predicted spread, result, and cover probability
        """
        home_rating = game_data.get("home_rating", 1500)
        away_rating = game_data.get("away_rating", 1500)
        
        predicted_margin = (home_rating - away_rating) * 0.05 + self.home_court_advantage
        
        # Compare against spread if available
        spread = game_data.get("spread", 0)
        
        cover_probability = self._sigmoid(predicted_margin / 5.0)
        
        result = {
            "predicted_margin": float(predicted_margin),
            "spread": spread,
            "home_cover_probability": float(cover_probability),
            "spread_result": "home" if predicted_margin > spread else "away",
            "confidence": float(abs(2 * cover_probability - 1)),
        }
        
        # Cache result
        game_id = game_data.get("game_id", "")
        cache_path = MODEL_CACHE_DIR / f"spread_{game_id}.json"
        with open(cache_path, "w") as f:
            json.dump(result, f)
        
        return result
    
    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "SpreadModel",
            "trained": self.trained,
            "home_court_advantage": self.home_court_advantage,
        }


class PlayerPropsModel:
    """Predicts player proposition statistics.
    
    Uses historical performance data to predict player props
    like points, rebounds, assists, and combined totals.
    """
    
    def __init__(self):
        self.trained = False
    
    def predict(self, player_data: dict) -> dict:
        """Predict player props.
        
        Args:
            player_data: Dictionary containing player stats and context
            
        Returns:
            Dict with predicted stats and probabilities
        """
        avg_pts = player_data.get("avg_points", 20.0)
        avg_reb = player_data.get("avg_rebounds", 5.0)
        avg_ast = player_data.get("avg_assists", 5.0)
        minutes = player_data.get("minutes", 30.0)
        opponent_def = player_data.get("opponent_defensive_rating", 110.0)
        
        # Adjust predictions based on context
        pace_factor = 100 / player_data.get("pace", 100)
        minutes_factor = minutes / 30.0
        
        predicted_pts = avg_pts * pace_factor * minutes_factor * (150 / opponent_def)
        predicted_reb = avg_reb * pace_factor * minutes_factor
        predicted_ast = avg_ast * pace_factor * minutes_factor
        
        # Calculate probabilities for over/under
        pts_over_prob = self._sigmoid((predicted_pts - 2.5) / 3.0)
        reb_over_prob = self._sigmoid((predicted_reb - 1.5) / 2.0)
        ast_over_prob = self._sigmoid((predicted_ast - 2.5) / 2.0)
        
        result = {
            "points": {"predicted": float(predicted_pts), "over_prob": float(pts_over_prob)},
            "rebounds": {"predicted": float(predicted_reb), "over_prob": float(reb_over_prob)},
            "assists": {"predicted": float(predicted_ast), "over_prob": float(ast_over_prob)},
        }
        
        # Cache result
        player_id = player_data.get("player_id", "")
        cache_path = MODEL_CACHE_DIR / f"player_props_{player_id}.json"
        with open(cache_path, "w") as f:
            json.dump(result, f)
        
        return result
    
    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {
            "model_type": "PlayerPropsModel",
            "trained": self.trained,
        }


class ManifestModel:
    """Generates game manifest data for the prediction API.
    
    Creates structured game manifests with all relevant information
    for the prediction pipeline.
    """
    
    def generate_manifest(self, game_data: dict) -> dict:
        """Generate a complete game manifest.
        
        Args:
            game_data: Dictionary containing game information
            
        Returns:
            Complete manifest dict with all game information
        """
        manifest = {
            "game_id": game_data.get("game_id", ""),
            "game_date": game_data.get("date", ""),
            "home_team": game_data.get("home_team", {}),
            "away_team": game_data.get("away_team", {}),
            "venue": game_data.get("venue", {}),
            "season": game_data.get("season", ""),
            "playoffs": game_data.get("playoffs", False),
            "status": game_data.get("status", "scheduled"),
        }
        
        # Cache manifest
        game_id = game_data.get("game_id", "")
        cache_path = MODEL_CACHE_DIR / f"manifest_{game_id}.json"
        with open(cache_path, "w") as f:
            json.dump(manifest, f)
        
        return manifest
    
    def get_model_info(self) -> dict:
        """Get model information."""
        return {"model_type": "ManifestModel"}