"""
ESPN data module for fetching NBA injury and lineup status.
Uses unofficial ESPN API endpoints.
"""

import json
import time
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from nba_predictor import config

# Base URL for the unofficial ESPN API
ESPN_API_BASE = "https://site.api.espn.com/apis/v2/sports/basketball"

# Cache directory for ESPN data
ESPN_CACHE_DIR = config.CACHE_DIR / "espn"


def _cache_path(endpoint: str, id_value: str) -> Path:
    """Generate cache file path for a given endpoint and ID."""
    # Sanitize the id_value for use in filename
    safe_id = id_value.replace("/", "_").replace(":", "_")
    return ESPN_CACHE_DIR / f"{endpoint}_{safe_id}.json"


def _load_cache(endpoint: str, id_value: str) -> dict | None:
    """Load cached data if it exists and is not stale."""
    cache_file = _cache_path(endpoint, id_value)
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                cached = json.load(f)
                return cached
        except (json.JSONDecodeError, IOError):
            return None
    return None


def _save_cache(endpoint: str, id_value: str, data: dict) -> None:
    """Save data to cache."""
    cache_file = _cache_path(endpoint, id_value)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(data, f, indent=2)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=32),
    reraise=True,
)
def _fetch_espn_data(url: str) -> dict:
    """
    Fetch data from ESPN API with exponential backoff retry.
    
    Args:
        url: The ESPN API URL to fetch
        
    Returns:
        JSON response as dict
        
    Raises:
        requests.exceptions.RequestException: If all retries fail
    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def get_injuries(game_id: str) -> dict:
    """
    Get injury status for a game from ESPN.
    
    Args:
        game_id: The ESPN game ID
        
    Returns:
        Dict containing injury information with keys:
            - gameId: The game ID
            - teamId: The team ID (if available)
            - injuries: List of injury objects
            
    Example response structure:
    {
        "gameId": "401784512",
        "teamId": 1610612738,
        "injuries": [
            {
                "athlete": {"id": "12345", "fullName": "Jayson Tatum", ...},
                "type": {"name": "DTD", "shortName": "DTD"},
                "status": "Out",
                "description": "Right knee injury"
            }
        ]
    }
    """
    # Check cache first
    cached = _load_cache("injuries", game_id)
    if cached is not None:
        return cached
    
    # Build ESPN API URL for injuries
    # The unofficial ESPN API returns injury data via scoreboard endpoint
    url = f"{ESPN_API_BASE}/scoreboard?event={game_id}"
    
    try:
        data = _fetch_espn_data(url)
        
        # Parse injury data from the response
        result = _parse_injuries_response(game_id, data)
        
        # Save to cache
        _save_cache("injuries", game_id, result)
        
        return result
        
    except requests.exceptions.RequestException:
        # Re-raise to trigger retry
        raise


def _parse_injuries_response(game_id: str, data: dict) -> dict:
    """
    Parse injury data from ESPN API response.
    
    Args:
        game_id: The game ID
        data: Raw API response data
        
    Returns:
        Parsed injuries response
    """
    result = {
        "gameId": game_id,
        "teamId": None,
        "injuries": []
    }
    
    # Try to extract team info from the response
    if "events" in data and len(data["events"]) > 0:
        event = data["events"][0]
        if "competitions" in event and len(event["competitions"]) > 0:
            competition = event["competitions"][0]
            if "competitors" in competition:
                for competitor in competition["competitors"]:
                    if "team" in competitor:
                        result["teamId"] = competitor["team"].get("id")
                    if "injuries" in competitor:
                        result["injuries"] = competitor["injuries"]
    
    return result


def get_lineup(game_id: str) -> dict:
    """
    Get lineup for a game from ESPN.
    
    Args:
        game_id: The ESPN game ID
        
    Returns:
        Dict containing lineup information with keys:
            - gameId: The game ID
            - teamId: The team ID (if available)
            - lineup: List of lineup objects
            
    Example response structure:
    {
        "gameId": "401784512",
        "teamId": 1610612738,
        "lineup": [
            {
                "athlete": {"id": "12345", "fullName": "Jayson Tatum", ...},
                "position": "SF",
                "status": "Active"
            }
        ]
    }
    """
    # Check cache first
    cached = _load_cache("lineup", game_id)
    if cached is not None:
        return cached
    
    # Build ESPN API URL for lineup
    # The unofficial ESPN API returns lineup data via scoreboard endpoint
    url = f"{ESPN_API_BASE}/scoreboard?event={game_id}"
    
    try:
        data = _fetch_espn_data(url)
        
        # Parse lineup data from the response
        result = _parse_lineup_response(game_id, data)
        
        # Save to cache
        _save_cache("lineup", game_id, result)
        
        return result
        
    except requests.exceptions.RequestException:
        # Re-raise to trigger retry
        raise


def _parse_lineup_response(game_id: str, data: dict) -> dict:
    """
    Parse lineup data from ESPN API response.
    
    Args:
        game_id: The game ID
        data: Raw API response data
        
    Returns:
        Parsed lineup response
    """
    result = {
        "gameId": game_id,
        "teamId": None,
        "lineup": []
    }
    
    # Try to extract team info from the response
    if "events" in data and len(data["events"]) > 0:
        event = data["events"][0]
        if "competitions" in event and len(event["competitions"]) > 0:
            competition = event["competitions"][0]
            if "competitors" in competition:
                for competitor in competition["competitors"]:
                    if "team" in competitor:
                        result["teamId"] = competitor["team"].get("id")
                    if "lineup" in competitor:
                        result["lineup"] = competitor["lineup"]
    
    return result


def get_team_status(team_id: int) -> dict:
    """
    Get team status including rest days and back-to-back info from ESPN.
    
    Args:
        team_id: The ESPN team ID
        
    Returns:
        Dict containing team status with keys:
            - teamId: The team ID
            - team: Team information dict
            - restDays: Number of rest days
            - backToBack: Whether it's a back-to-back
            - lastGameDate: Date of last game (ISO format)
            - nextGameDate: Date of next game (ISO format)
            
    Example response structure:
    {
        "teamId": 1610612738,
        "team": {
            "id": 1610612738,
            "location": "Boston",
            "name": "Celtics",
            "abbreviation": "BOS"
        },
        "restDays": 2,
        "backToBack": False,
        "lastGameDate": "2024-04-01",
        "nextGameDate": "2024-04-04"
    }
    """
    # Check cache first
    cached = _load_cache("teamStatus", str(team_id))
    if cached is not None:
        return cached
    
    # Build ESPN API URL for team status
    url = f"{ESPN_API_BASE}/teams/{team_id}/schedule"
    
    try:
        data = _fetch_espn_data(url)
        
        # Parse team status data from the response
        result = _parse_team_status_response(team_id, data)
        
        # Save to cache
        _save_cache("teamStatus", str(team_id), result)
        
        return result
        
    except requests.exceptions.RequestException:
        # Re-raise to trigger retry
        raise


def _parse_team_status_response(team_id: int, data: dict) -> dict:
    """
    Parse team status data from ESPN API response.
    
    Args:
        team_id: The team ID
        data: Raw API response data
        
    Returns:
        Parsed team status response
    """
    result = {
        "teamId": team_id,
        "team": {},
        "restDays": 0,
        "backToBack": False,
        "lastGameDate": None,
        "nextGameDate": None
    }
    
    # Extract team info
    if "team" in data:
        team_info = data["team"]
        result["team"] = {
            "id": team_info.get("id"),
            "location": team_info.get("location"),
            "name": team_info.get("name"),
            "abbreviation": team_info.get("abbreviation")
        }
    
    # Extract schedule info for rest days and back-to-back
    if "events" in data:
        events = data["events"]
        if len(events) > 0:
            # Get last completed game
            completed_events = [e for e in events if e.get("status", {}).get("type", {}).get("name") == "STATUS_FINAL"]
            if len(completed_events) > 0:
                last_game = completed_events[-1]
                result["lastGameDate"] = last_game.get("date")
                
                # Calculate rest days (simplified)
                # In a real implementation, this would parse dates properly
                result["restDays"] = 1  # Default assumption
                
            # Get next scheduled game
            scheduled_events = [e for e in events if e.get("status", {}).get("type", {}).get("name") == "STATUS_SCHEDULED"]
            if len(scheduled_events) > 0:
                next_game = scheduled_events[0]
                result["nextGameDate"] = next_game.get("date")
                
                # Check for back-to-back
                if result["lastGameDate"] and result["nextGameDate"]:
                    result["backToBack"] = True  # Simplified logic
    
    return result
