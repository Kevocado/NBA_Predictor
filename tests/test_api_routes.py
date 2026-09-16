"""Tests for api/routes.py."""
import pytest
from unittest.mock import MagicMock, patch
import json
from pathlib import Path


class TestGetGames:
    """Test suite for get_games."""

    def test_returns_list(self):
        """Test get_games returns list."""
        from nba_predictor.api.routes import get_games
        
        with patch("nba_predictor.api.routes.nba_api") as mock_api:
            mock_api.get_schedule.return_value = [{"id": "0012400001"}, {"id": "0012400002"}]
            result = get_games("2024-10-25")
            
        assert isinstance(result, list)
        assert len(result) == 2

    def test_with_date(self):
        """Test get_games with specific date."""
        from nba_predictor.api.routes import get_games
        
        with patch("nba_predictor.api.routes.nba_api") as mock_api:
            mock_api.get_schedule.return_value = [{"id": "0012400001"}]
            result = get_games("2024-10-25")
            
        assert isinstance(result, list)


class TestGetGame:
    """Test suite for get_game."""

    def test_returns_dict(self):
        """Test get_game returns dict."""
        from nba_predictor.api.routes import get_game
        
        with patch("nba_predictor.api.routes.nba_api") as mock_api:
            mock_api.get_schedule.return_value = [{"id": "0012400001", "home_team": "BOS"}]
            result = get_game("0012400001")
            
        assert isinstance(result, dict)
        assert result["id"] == "0012400001"

    def test_not_found(self):
        """Test get_game with nonexistent game."""
        from nba_predictor.api.routes import get_game
        
        with patch("nba_predictor.api.routes.nba_api") as mock_api:
            mock_api.get_schedule.return_value = []
            result = get_game("999999")
            
        assert isinstance(result, dict)
        assert result == {}


class TestGetTeams:
    """Test suite for get_teams."""

    def test_returns_list(self):
        """Test get_teams returns list."""
        from nba_predictor.api.routes import get_teams
        
        result = get_teams()
        
        assert isinstance(result, list)


class TestGetTeam:
    """Test suite for get_team."""

    def test_returns_dict(self):
        """Test get_team returns dict."""
        from nba_predictor.api.routes import get_team
        
        result = get_team(1610612738)
        
        assert isinstance(result, dict)

    def test_not_found(self):
        """Test get_team with nonexistent team."""
        from nba_predictor.api.routes import get_team
        
        result = get_team(999999)
        
        assert result == {}


class TestManifest:
    """Test suite for manifest endpoints."""

    def test_get_manifest(self):
        """Test get_manifest returns dict."""
        from nba_predictor.api.routes import get_manifest
        
        with patch("nba_predictor.api.routes.get_game") as mock_game:
            mock_game.return_value = {"game_id": "0012400001", "home_team": {}}
            with patch("nba_predictor.api.routes.ManifestModel") as mock_model_class:
                mock_model = MagicMock()
                mock_model.generate_manifest.return_value = {"game_id": "0012400001"}
                mock_model_class.return_value = mock_model
                
                result = get_manifest("0012400001")
                
            assert isinstance(result, dict)

    def test_get_manifests(self):
        """Test get_manifests returns list."""
        from nba_predictor.api.routes import get_manifests
        
        with patch("nba_predictor.api.routes.nba_api") as mock_api:
            mock_api.get_schedule.return_value = [{"id": "001"}, {"id": "002"}]
            with patch("nba_predictor.api.routes.ManifestModel") as mock_model_class:
                mock_model = MagicMock()
                mock_model.generate_manifest.return_value = {"game_id": "test"}
                mock_model_class.return_value = mock_model
                
                result = get_manifests()
                
            assert isinstance(result, list)
            assert len(result) == 2


class TestHubEndpoints:
    """Test suite for hub endpoints."""

    def test_get_hub_schedule(self):
        """Test get_hub_schedule."""
        from nba_predictor.api.routes import get_hub_schedule
        
        with patch("nba_predictor.api.routes.nba_api") as mock_api:
            mock_api.get_schedule.return_value = [{"id": "001"}]
            result = get_hub_schedule()
            
        assert isinstance(result, list)

    def test_get_hub_odds(self):
        """Test get_hub_odds."""
        from nba_predictor.api.routes import get_hub_odds
        
        with patch("nba_predictor.api.routes.balldontlie") as mock_api:
            mock_api.get_schedule.return_value = [{"id": "001"}]
            result = get_hub_odds()
            
        assert isinstance(result, list)

    def test_get_hub_injuries(self):
        """Test get_hub_injuries."""
        from nba_predictor.api.routes import get_hub_injuries
        
        with patch("nba_predictor.api.routes.espn") as mock_api:
            mock_api.get_injuries.return_value = [{"id": 1}]
            result = get_hub_injuries()
            
        assert isinstance(result, list)

    def test_get_hub_player_stats(self):
        """Test get_hub_player_stats."""
        from nba_predictor.api.routes import get_hub_player_stats
        
        with patch("nba_predictor.api.routes.balldontlie") as mock_api:
            mock_api.get_player_stats.return_value = {"id": "203999"}
            result = get_hub_player_stats()
            
        assert isinstance(result, list)

    def test_get_hub_lineups(self):
        """Test get_hub_lineups."""
        from nba_predictor.api.routes import get_hub_lineups
        
        with patch("nba_predictor.api.routes.espn") as mock_api:
            mock_api.get_lineup.return_value = {"gameId": "1"}
            result = get_hub_lineups()
            
        assert isinstance(result, list)


class TestPredictions:
    """Test suite for predictions."""

    def test_get_predictions(self):
        """Test get_predictions returns list."""
        from nba_predictor.api.routes import get_predictions
        
        games = [{"game_id": "001", "home": {"net_rating": 5}, "away": {"net_rating": 2}}]
        
        with patch("nba_predictor.api.routes.GameOutcomeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.predict.return_value = {"prediction": "home", "probability": 0.6}
            mock_model_class.return_value = mock_model
            
            result = get_predictions(games)
            
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_predictions_with_empty_games(self):
        """Test get_predictions with no games."""
        from nba_predictor.api.routes import get_predictions
        
        result = get_predictions([])
        
        assert isinstance(result, list)
        assert len(result) == 0


class TestValueBetsAPI:
    """Test suite for value bets API."""

    def test_get_value_bets(self):
        """Test get_value_bets returns list."""
        from nba_predictor.api.routes import get_value_bets
        
        games = [{"id": "001", "home_team": "BOS", "away_team": "LAL", "home_prob": 0.6}]
        
        with patch("nba_predictor.api.routes.detect_value_bets") as mock_detect:
            mock_detect.return_value = [{"game_id": "001", "edge": 0.1}]
            result = get_value_bets(games)
            
        assert isinstance(result, list)
        assert len(result) == 1

    def test_get_value_bets_empty(self):
        """Test get_value_bets with no value bets."""
        from nba_predictor.api.routes import get_value_bets
        
        games = [{"id": "001", "home_team": "BOS", "away_team": "LAL", "home_prob": 0.5}]
        
        with patch("nba_predictor.api.routes.detect_value_bets") as mock_detect:
            mock_detect.return_value = []
            result = get_value_bets(games)
            
        assert isinstance(result, list)
        assert len(result) == 0


class TestAllFeatures:
    """Test suite for get_all_features."""

    def test_returns_list(self):
        """Test get_all_features returns list."""
        from nba_predictor.api.routes import get_all_features
        
        games = [{"game_id": "001"}]
        team_data = {"recent_games": []}
        injury_data = {"injuries": []}
        
        with patch("nba_predictor.api.routes.build_features") as mock_build:
            mock_build.return_value = {"game_id": "001"}
            result = get_all_features(games, team_data, injury_data)
            
        assert isinstance(result, list)
        assert len(result) == 1


class TestAllSpreadPredictions:
    """Test suite for get_all_spread_predictions."""

    def test_returns_list(self):
        """Test get_all_spread_predictions returns list."""
        from nba_predictor.api.routes import get_all_spread_predictions
        
        games = [{"game_id": "001", "home_rating": 1500, "away_rating": 1490}]
        
        with patch("nba_predictor.api.routes.SpreadModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.predict.return_value = {"predicted_margin": 5.0}
            mock_model_class.return_value = mock_model
            
            result = get_all_spread_predictions(games)
            
        assert isinstance(result, list)
        assert len(result) == 1


class TestRoutesIntegration:
    """Integration tests for API routes."""

    def test_full_pipeline(self):
        """Test complete prediction pipeline."""
        from nba_predictor.api.routes import (
            get_games,
            get_predictions,
            get_value_bets,
            get_manifests,
        )
        
        # Test that all endpoints return valid types
        with patch("nba_predictor.api.routes.nba_api") as mock_api:
            mock_api.get_schedule.return_value = [
                {"id": "001", "home_team": "BOS", "away_team": "LAL"}
            ]
            
            games = get_games("2024-10-25")
            assert isinstance(games, list)
            
            with patch("nba_predictor.api.routes.GameOutcomeModel") as mock_model_class:
                mock_model = MagicMock()
                mock_model.predict.return_value = {"prediction": "home", "probability": 0.6}
                mock_model_class.return_value = mock_model
                
                predictions = get_predictions(games)
                assert isinstance(predictions, list)
            
            with patch("nba_predictor.api.routes.detect_value_bets") as mock_detect:
                mock_detect.return_value = [{"edge": 0.1}]
                bets = get_value_bets(games)
                assert isinstance(bets, list)
            
            manifests = get_manifests()
            assert isinstance(manifests, list)