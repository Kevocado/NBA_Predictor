"""Extended API routes for NBA predictions."""
from typing import Any
from nba_predictor.data import nba_api, balldontlie, injuries, espn
from nba_predictor.models import GameOutcomeModel, SpreadModel, PlayerPropsModel, ManifestModel
from nba_predictor.features.build import build_features
from nba_predictor.odds.value_bets import detect_value_bets, compute_value_bets

# Extended routes mapping
ROUTES = {
    "GET /games": "Get all games for a date",
    "GET /games/{game_id}": "Get specific game details",
    "GET /teams": "Get all team data",
    "GET /teams/{team_id}": "Get specific team data",
    "GET /manifest": "Get all game manifests",
    "GET /manifest/{game_id}": "Get game manifest",
    "GET /hub/schedule": "Get schedule from hub",
    "GET /hub/odds": "Get odds from hub",
    "GET /hub/injuries": "Get injuries from hub",
    "GET /hub/player_stats": "Get player stats from hub",
    "GET /hub/lineups": "Get lineups from hub",
    "POST /predictions": "Get predictions for games",
    "POST /value_bets": "Detect value bets",
    "GET /health": "Health check",
}


def get_games(date: str) -> list[dict]:
    """Get all games for a date."""
    return nba_api.get_schedule(date)


def get_game(game_id: str) -> dict:
    """Get specific game details."""
    schedule = nba_api.get_schedule("2024-10-25")
    for game in schedule:
        if str(game.get("id", "")) == str(game_id):
            return game
    return {}


def get_teams() -> list[dict]:
    """Get all team data."""
    from nba_predictor.data.team_reference import TEAMS
    return [team.__dict__ for team in TEAMS]


def get_team(team_id: int) -> dict:
    """Get specific team data."""
    from nba_predictor.data.team_reference import TEAMS
    for team in TEAMS:
        if team.nba_api_id == team_id:
            return team.__dict__
    return {}


def get_manifest(game_id: str) -> dict:
    """Get game manifest."""
    game_data = get_game(game_id)
    if not game_data:
        return {}
    
    manifest = ManifestModel()
    return manifest.generate_manifest(game_data)


def get_manifests() -> list[dict]:
    """Get all game manifests."""
    schedule = nba_api.get_schedule("2024-10-25")
    manifests = []
    manifest_model = ManifestModel()
    for game in schedule:
        manifests.append(manifest_model.generate_manifest(game))
    return manifests


def get_hub_schedule() -> list[dict]:
    """Get schedule from hub data."""
    return nba_api.get_schedule("2024-10-25")


def get_hub_odds() -> list[dict]:
    """Get odds from hub data."""
    return balldontlie.get_schedule("2024-10-25")


def get_hub_injuries() -> list[dict]:
    """Get injuries from hub data."""
    return espn.get_injuries("1")


def get_hub_player_stats() -> list[dict]:
    """Get player stats from hub data."""
    return [balldontlie.get_player_stats(203999, 2023)]


def get_hub_lineups() -> list[dict]:
    """Get lineups from hub data."""
    return [espn.get_lineup("1")]


def get_predictions(games: list[dict]) -> list[dict]:
    """Get predictions for games."""
    model = GameOutcomeModel()
    predictions = []
    for game in games:
        prediction = model.predict(game)
        predictions.append(prediction)
    return predictions


def get_value_bets(games: list[dict]) -> list[dict]:
    """Detect value bets for games."""
    value_bets = []
    for game in games:
        odds = {"home_odds": -110, "away_odds": -110}
        game_data = {
            "game_id": game.get("id", ""),
            "home_team": game.get("home_team", {}),
            "away_team": game.get("away_team", {}),
            "home_prob": 0.5,
        }
        bets = detect_value_bets(game_data, odds)
        value_bets.extend(bets)
    return value_bets


def get_all_features(games: list[dict], team_data: dict, injury_data: dict) -> list[dict]:
    """Build features for all games."""
    features_list = []
    for game in games:
        features = build_features(game, team_data, injury_data)
        features_list.append(features)
    return features_list


def get_all_spread_predictions(games: list[dict]) -> list[dict]:
    """Get spread predictions for all games."""
    model = SpreadModel()
    predictions = []
    for game in games:
        prediction = model.predict(game)
        predictions.append(prediction)
    return predictions