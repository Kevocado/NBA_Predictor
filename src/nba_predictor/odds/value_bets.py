"""Value bet detection using the Shin two-way model.

Implements the Shin (2005) model for detecting value bets
in NBA odds markets.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
from nba_predictor import config
from nba_predictor.models import GameOutcomeModel, SpreadModel

# Cache directory for value bets
VALUE_BETS_CACHE_DIR = config.CACHE_DIR / "value_bets"
VALUE_BETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def shin_two_way(p_home: float, p_away: float, m: float = 0.0) -> dict:
    """Shin two-way model to calculate fair probabilities.
    
    Args:
        p_home: Market probability for home team (0-1)
        p_away: Market probability for away team (0-1)
        m: Margin parameter (default 0 for no margin)
        
    Returns:
        Dict with fair probabilities and value bet flags
    """
    # Convert to log-odds
    def _log_odds(p: float) -> float:
        return np.log(p / (1 - p))
    
    def _expit(x: float) -> float:
        """Inverse logit."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    try:
        o_h = _log_odds(p_home)
        o_a = _log_odds(p_away)
    except (ValueError, ZeroDivisionError):
        return {"error": "Invalid probabilities"}
    
    # Calculate adjusted probabilities using Shin's model
    # Without margin:
    # p_i = exp(o_i - m) / (sum_j exp(o_j - m))
    
    # Simplified Shin two-way
    if m == 0:
        sum_exp = np.exp(o_h) + np.exp(o_a)
        fair_home = np.exp(o_h) / sum_exp
        fair_away = np.exp(o_a) / sum_exp
    else:
        fair_home = np.exp(o_h - m) / (np.exp(o_h - m) + np.exp(o_a - m))
        fair_away = np.exp(o_a - m) / (np.exp(o_h - m) + np.exp(o_a - m))
    
    return {
        "fair_prob_home": float(fair_home),
        "fair_prob_away": float(fair_away),
        "market_prob_home": float(p_home),
        "market_prob_away": float(p_away),
        "margin": float(m),
    }


def detect_value_bets(game_data: dict, odds_data: dict, threshold: float = 0.05) -> list[dict]:
    """Detect value bets using the Shin two-way model.
    
    Args:
        game_data: Dictionary containing game information and predicted probabilities
        odds_data: Dictionary containing market odds
        threshold: Minimum edge required to flag as value bet
        
    Returns:
        List of detected value bets
    """
    # Get predicted probability from game data
    predicted_prob = game_data.get("home_prob", 0.5)
    
    # Get market implied probability from odds
    # Convert American odds to implied probability
    def _american_to_implied(odds: float) -> float:
        if odds > 0:
            return odds / (odds + 100)
        else:
            return 100 / (abs(odds) + 100)
    
    home_odds = odds_data.get("home_odds", 0)
    away_odds = odds_data.get("away_odds", 0)
    
    if home_odds == 0 or away_odds == 0:
        return []
    
    market_prob_home = _american_to_implied(home_odds)
    market_prob_away = _american_to_implied(away_odds)
    
    # Normalize market probabilities
    total = market_prob_home + market_prob_away
    if total == 0:
        return []
    market_prob_home /= total
    market_prob_away /= total
    
    # Apply Shin two-way model
    result = shin_two_way(market_prob_home, market_prob_away)
    
    fair_prob_home = result.get("fair_prob_home", market_prob_home)
    
    # Calculate edge
    edge = predicted_prob - fair_prob_home
    
    value_bets = []
    
    if abs(edge) > threshold:
        bet = {
            "game_id": game_data.get("game_id", ""),
            "home_team": game_data.get("home_team", ""),
            "away_team": game_data.get("away_team", ""),
            "predicted_prob": float(predicted_prob),
            "fair_prob": float(fair_prob_home),
            "edge": float(edge),
            "recommendation": "home" if edge > 0 else "away",
            "confidence": float(abs(edge)),
        }
        value_bets.append(bet)
    
    # Cache results
    game_id = game_data.get("game_id", "")
    cache_path = VALUE_BETS_CACHE_DIR / f"value_bets_{game_id}.json"
    with open(cache_path, "w") as f:
        json.dump(value_bets, f)
    
    return value_bets


def compute_value_bets(game_data: dict, odds_data: dict) -> dict:
    """Compute all value bets for a game.
    
    Args:
        game_data: Dictionary containing game information and predictions
        odds_data: Dictionary containing market odds
        
    Returns:
        Dict with all value bet calculations
    """
    predicted_prob = game_data.get("home_prob", 0.5)
    market_prob_home = odds_data.get("home_implied_prob", 0.5)
    market_prob_away = odds_data.get("away_implied_prob", 0.5)
    
    # Normalize
    total = market_prob_home + market_prob_away
    if total > 0:
        market_prob_home /= total
        market_prob_away /= total
    
    # Apply Shin two-way
    shin_result = shin_two_way(market_prob_home, market_prob_away)
    fair_prob = shin_result.get("fair_prob_home", market_prob_home)
    
    # Home value bet
    home_edge = predicted_prob - fair_prob
    home_value = abs(home_edge) > 0.05
    
    # Away value bet
    away_fair_prob = 1 - fair_prob
    away_edge = (1 - predicted_prob) - away_fair_prob
    away_value = abs(away_edge) > 0.05
    
    return {
        "game_id": game_data.get("game_id", ""),
        "home_value_bet": {
            "exists": home_value,
            "edge": float(home_edge),
            "predicted_prob": float(predicted_prob),
            "fair_prob": float(fair_prob),
        },
        "away_value_bet": {
            "exists": away_value,
            "edge": float(away_edge),
            "predicted_prob": float(1 - predicted_prob),
            "fair_prob": float(away_fair_prob),
        },
        "shin_fair_probs": {
            "home": float(fair_prob),
            "away": float(away_fair_prob),
        },
    }


def get_value_bets(game_id: Optional[str] = None) -> list[dict]:
    """Retrieve cached value bets."""
    if game_id:
        cache_path = VALUE_BETS_CACHE_DIR / f"value_bets_{game_id}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return []
        return []
    
    # Return all cached value bets
    results = []
    for cache_file in VALUE_BETS_CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file, "r") as f:
                results.extend(json.load(f))
        except (json.JSONDecodeError, IOError):
            continue
    
    return results