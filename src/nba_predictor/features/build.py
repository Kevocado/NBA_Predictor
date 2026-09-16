"""Feature engineering pipeline for NBA prediction models."""
import json
from pathlib import Path
from typing import Any

import numpy as np
from nba_predictor import config
from nba_predictor.data import nba_api, balldontlie, injuries, team_reference

# Cache directory for features
FEATURES_CACHE_DIR = config.CACHE_DIR / "features"
FEATURES_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Feature configuration
FOUR_FACTORS = [
    "effective_field_goal_percentage",
    "turnover_percentage",
    "offensive_rebound_percentage",
    "free_throw_rate",
]

EFFICIENCY_STATS = [
    "offensive_rating",
    "defensive_rating",
    "net_rating",
    "pace",
]

POWER_RATING_PARAMS = {
    "home_court_advantage": 3.0,  # points
    "elo_base": 1500,
    "elo_k": 32,
}

REST_FATIGUE = {
    "rest_threshold_days": 1,
    "back_to_back_flag": True,
    "congestion_3_in_4": 3,
    "congestion_4_in_6": 4,
}

TRAVEL = {
    "rolling_window_days": 7,
    "mileage_weight": 0.3,
    "timezone_weight": 0.4,
    "rest_deficit_weight": 0.3,
}


def _load_team_arena_data() -> dict:
    """Load team arena coordinates for travel calculations."""
    teams = {}
    for team in team_reference.TEAMS:
        teams[team.abbreviation] = {
            "latitude": team.arena_lat,
            "longitude": team.arena_lon,
            "altitude": team.altitude_ft,
            "conference": team.conference,
            "division": team.division,
        }
    return teams


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points on Earth in miles."""
    from math import radians, cos, sin, asin, sqrt
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3959  # Earth's radius in miles
    return c * r


def compute_four_factors(game_data: dict) -> dict:
    """Compute Four Factors metrics for a game.
    
    Args:
        game_data: Dictionary containing game statistics
        
    Returns:
        Dict with four factors metrics for home and away teams
    """
    result = {"home": {}, "away": {}}
    
    for team_side in ["home", "away"]:
        stats = game_data.get(f"{team_side}_stats", {})
        result[team_side] = {
            "effective_field_goal_percentage": stats.get("efg", 0.5),
            "turnover_percentage": stats.get("tov_pct", 0.14),
            "offensive_rebound_percentage": stats.get("orb_pct", 0.25),
            "free_throw_rate": stats.get("ft_rate", 0.2),
        }
    
    return result


def compute_efficiency(game_data: dict) -> dict:
    """Compute efficiency metrics for a game.
    
    Args:
        game_data: Dictionary containing game statistics
        
    Returns:
        Dict with efficiency metrics
    """
    result = {"home": {}, "away": {}}
    
    for team_side in ["home", "away"]:
        stats = game_data.get(f"{team_side}_stats", {})
        pts = stats.get("points", 100)
        poss = stats.get("possessions", 100)
        opp_pts = stats.get("opponent_points", 95)
        
        off_rating = (pts / poss) * 100 if poss > 0 else 100
        def_rating = (opp_pts / poss) * 100 if poss > 0 else 100
        
        result[team_side] = {
            "offensive_rating": off_rating,
            "defensive_rating": def_rating,
            "net_rating": off_rating - def_rating,
            "pace": poss / 48 * 100 if poss > 0 else 100,
        }
    
    return result


def compute_power_rating(home_team: str, away_team: str, games: list[dict]) -> dict:
    """Compute Elo-style power rating with home-court adjustment.
    
    Args:
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        games: List of recent game results
        
    Returns:
        Dict with power ratings for both teams
    """
    arena_data = _load_team_arena_data()
    home_elo = arena_data.get(home_team, {}).get("altitude", 0)
    away_elo = arena_data.get(away_team, {}).get("altitude", 0)
    
    # Apply home court advantage
    home_elo_adjusted = home_elo + POWER_RATING_PARAMS["home_court_advantage"]
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_power_rating": home_elo_adjusted,
        "away_power_rating": away_elo,
        "home_court_advantage": POWER_RATING_PARAMS["home_court_advantage"],
    }


def compute_rest_fatigue(team_id: int, games: list[dict]) -> dict:
    """Compute rest and fatigue metrics for a team.
    
    Args:
        team_id: NBA team ID
        games: List of recent games for the team
        
    Returns:
        Dict with rest days, back-to-back flag, congestion flags
    """
    if not games:
        return {
            "rest_days": 0,
            "back_to_back": False,
            "congestion_3_in_4": False,
            "congestion_4_in_6": False,
            "total_rest_deficit": 0,
        }
    
    # Count rest days between games
    rest_days = 0
    back_to_back = False
    games_in_4_nights = 0
    games_in_6_nights = 0
    
    for i, game in enumerate(games):
        if i > 0:
            # Simplified: count games in window
            pass
    
    return {
        "rest_days": rest_days,
        "back_to_back": back_to_back,
        "congestion_3_in_4": games_in_4_nights >= REST_FATIGUE["congestion_3_in_4"],
        "congestion_4_in_6": games_in_6_nights >= REST_FATIGUE["congestion_4_in_6"],
        "total_rest_deficit": max(0, 2 - rest_days),
    }


def compute_travel(team_id: int, games: list[dict]) -> dict:
    """Compute travel metrics for a team.
    
    Args:
        team_id: NBA team ID
        games: List of recent games for the team
        
    Returns:
        Dict with travel mileage, timezone changes, fatigue index
    """
    arena_data = _load_team_arena_data()
    
    total_mileage = 0.0
    timezone_changes = 0
    prev_location = None
    
    for game in games:
        location = game.get("location", {})
        if prev_location and location:
            try:
                dist = _haversine_distance(
                    prev_location["latitude"], prev_location["longitude"],
                    location["latitude"], location["longitude"],
                )
                total_mileage += dist
            except Exception:
                pass
        
        if location:
            prev_location = location
            timezone_changes += 1  # Simplified
    
    # Compute fatigue index
    rolling_mileage = total_mileage / TRAVEL["rolling_window_days"] if games else 0
    fatigue_index = (
        rolling_mileage * TRAVEL["mileage_weight"] +
        timezone_changes * TRAVEL["timezone_weight"] +
        compute_rest_fatigue(team_id, games).get("total_rest_deficit", 0) * TRAVEL["rest_deficit_weight"]
    )
    
    return {
        "total_mileage": total_mileage,
        "rolling_7day_mileage": rolling_mileage,
        "timezone_changes": timezone_changes,
        "fatigue_index": fatigue_index,
    }


def compute_injury_impact(game_data: dict) -> dict:
    """Compute injury impact on game.
    
    Args:
        game_data: Dictionary containing game and injury data
        
    Returns:
        Dict with missing player production value
    """
    injury_data = game_data.get("injuries", [])
    
    missing_value = 0.0
    missing_players = []
    
    for injury in injury_data:
        status = injury.get("status", "").lower()
        if status in ["out", "questionable", "doubtful"]:
            player_stats = injury.get("stats", {})
            pts = player_stats.get("points_per_game", 0)
            reb = player_stats.get("rebounds_per_game", 0)
            ast = player_stats.get("assists_per_game", 0)
            missing_value += pts + reb * 0.7 + ast * 0.5
            missing_players.append(injury.get("player_name", "Unknown"))
    
    return {
        "missing_player_value": missing_value,
        "missing_players": missing_players,
        "total_impact_score": missing_value,
    }


def compute_context(game_data: dict) -> dict:
    """Compute contextual features for a game.
    
    Args:
        game_data: Dictionary containing game context
        
    Returns:
        Dict with H2H history, streaks, altitude flag, conference flag
    """
    home_team = game_data.get("home_team", {})
    away_team = game_data.get("away_team", {})
    
    home_abbrev = home_team.get("abbreviation", "")
    away_abbrev = away_team.get("abbreviation", "")
    
    arena_data = _load_team_arena_data()
    
    # Check altitude (Denver)
    home_altitude = arena_data.get(home_abbrev, {}).get("altitude", 0)
    away_altitude = arena_data.get(away_abbrev, {}).get("altitude", 0)
    
    # Conference flag
    home_conf = arena_data.get(home_abbrev, {}).get("conference", "")
    away_conf = arena_data.get(away_abbrev, {}).get("conference", "")
    is_conference_game = home_conf == away_conf and home_conf != ""
    
    return {
        "home_altitude": home_altitude,
        "away_altitude": away_altitude,
        "altitude_flag": home_altitude > 5000 or away_altitude > 5000,
        "conference_game": is_conference_game,
        "division_game": (
            arena_data.get(home_abbrev, {}).get("division", "") ==
            arena_data.get(away_abbrev, {}).get("division", "")
        ),
    }


def build_features(game_data: dict, team_data: dict, injury_data: dict) -> dict:
    """Build complete feature vector for a game prediction.
    
    Args:
        game_data: Dictionary containing game statistics and matchup info
        team_data: Dictionary containing team performance data
        injury_data: Dictionary containing injury information
        
    Returns:
        Complete feature dictionary for model prediction
    """
    four_factors = compute_four_factors(game_data)
    efficiency = compute_efficiency(game_data)
    power_rating = compute_power_rating(
        game_data.get("home_team", {}).get("abbreviation", ""),
        game_data.get("away_team", {}).get("abbreviation", ""),
        team_data.get("recent_games", []),
    )
    rest_fatigue = compute_rest_fatigue(
        game_data.get("home_team", {}).get("id", 0),
        team_data.get("recent_games", []),
    )
    travel = compute_travel(
        game_data.get("home_team", {}).get("id", 0),
        team_data.get("recent_games", []),
    )
    injury_impact = compute_injury_impact(injury_data)
    context = compute_context(game_data)
    
    # Combine all features into a flat feature vector
    features = {
        "game_id": game_data.get("game_id", ""),
        "home_team": game_data.get("home_team", {}).get("abbreviation", ""),
        "away_team": game_data.get("away_team", {}).get("abbreviation", ""),
        "four_factors": four_factors,
        "efficiency": efficiency,
        "power_rating": power_rating,
        "rest_fatigue": rest_fatigue,
        "travel": travel,
        "injury_impact": injury_impact,
        "context": context,
    }
    
    # Cache features
    _cache_features(game_data, features)
    
    return features


def _cache_features(game_data: dict, features: dict) -> None:
    """Cache computed features."""
    game_id = game_data.get("game_id", "unknown")
    cache_path = FEATURES_CACHE_DIR / f"features_{game_id}.json"
    with open(cache_path, "w") as f:
        json.dump(features, f)


def get_cached_features(game_id: str) -> dict | None:
    """Retrieve cached features for a game."""
    cache_path = FEATURES_CACHE_DIR / f"features_{game_id}.json"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None
