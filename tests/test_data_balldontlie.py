"""Tests for the balldontlie data module."""
import pytest
from unittest.mock import patch, Mock
from pathlib import Path
import json
import shutil


# Cache directory for tests
TEST_CACHE_DIR = Path("/tmp/balldontlie_test_cache")


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the test cache before each test."""
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)
    TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Patch config.CACHE_DIR to use test cache directory
    with patch("nba_predictor.data.balldontlie.config.CACHE_DIR", TEST_CACHE_DIR):
        yield

    # Cleanup after test
    if TEST_CACHE_DIR.exists():
        shutil.rmtree(TEST_CACHE_DIR)


class TestGetSchedule:
    """Tests for get_schedule function."""

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_schedule_success(self, mock_api_key):
        """Test successful schedule retrieval."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {
            "data": [
                {
                    "id": 12345,
                    "date": "2024-01-15T00:00:00Z",
                    "home_team_id": 1610612738,
                    "visitor_team_id": 1610612751,
                }
            ]
        }

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_schedule("2024-01-15")

            assert isinstance(result, list)
            assert len(result) == 1
            assert result[0]["id"] == 12345

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_schedule_empty(self, mock_api_key):
        """Test schedule with no games."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {"data": []}

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_schedule("2024-01-15")

            assert isinstance(result, list)
            assert len(result) == 0

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_schedule_rate_limit(self, mock_api_key):
        """Test rate limit handling (429 response)."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.raise_for_status = Mock(side_effect=Exception("Rate limit exceeded"))
            mock_get.return_value = mock_response

            with pytest.raises(Exception, match="Rate limit exceeded"):
                balldontlie.get_schedule("2024-01-15")


class TestGetBoxscore:
    """Tests for get_boxscore function."""

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_boxscore_success(self, mock_api_key):
        """Test successful boxscore retrieval."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {
            "id": 12345,
            "game_date": "2024-01-15",
            "home_team_id": 1610612738,
            "visitor_team_id": 1610612751,
            "home_team_score": 110,
            "visitor_team_score": 105,
        }

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_boxscore(12345)

            assert isinstance(result, dict)
            assert result["id"] == 12345
            assert result["home_team_score"] == 110

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_boxscore_not_found(self, mock_api_key):
        """Test boxscore for non-existent game."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.raise_for_status = Mock(side_effect=Exception("Not found"))
            mock_get.return_value = mock_response

            with pytest.raises(Exception, match="Not found"):
                balldontlie.get_boxscore(99999)


class TestGetPlayerStats:
    """Tests for get_player_stats function."""

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_stats_success(self, mock_api_key):
        """Test successful player stats retrieval."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {
            "data": [
                {
                    "id": 456,
                    "player_id": 203999,
                    "season": 2023,
                    "games_played": 70,
                    "pts": 25.5,
                    "reb": 7.2,
                    "ast": 5.8,
                }
            ]
        }

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_player_stats(203999, 2023)

            assert isinstance(result, dict)
            assert result["player_id"] == 203999
            assert result["season"] == 2023
            assert result["pts"] == 25.5

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_stats_no_data(self, mock_api_key):
        """Test player stats when no data returned."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {"data": []}

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_player_stats(999999, 2023)

            assert isinstance(result, dict)
            assert result == {}


class TestGetTeamStats:
    """Tests for get_team_stats function."""

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_team_stats_success(self, mock_api_key):
        """Test successful team stats retrieval."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {
            "data": [
                {
                    "id": 789,
                    "team_id": 1610612738,
                    "season": 2023,
                    "games_played": 82,
                    "wins": 64,
                    "losses": 18,
                    "pts_per_game": 118.5,
                }
            ]
        }

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_team_stats(1610612738, 2023)

            assert isinstance(result, dict)
            assert result["team_id"] == 1610612738
            assert result["season"] == 2023
            assert result["wins"] == 64

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_team_stats_not_found(self, mock_api_key):
        """Test team stats when team not found."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {"data": []}

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_team_stats(999999, 2023)

            assert isinstance(result, dict)
            assert result == {}


class TestGetPlayerGameLog:
    """Tests for get_player_game_log function."""

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_game_log_success(self, mock_api_key):
        """Test successful game log retrieval."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {
            "data": [
                {
                    "id": 1,
                    "player_id": 203999,
                    "game_id": 12345,
                    "date": "2024-01-15",
                    "pts": 30,
                    "reb": 8,
                    "ast": 6,
                },
                {
                    "id": 2,
                    "player_id": 203999,
                    "game_id": 12346,
                    "date": "2024-01-16",
                    "pts": 28,
                    "reb": 5,
                    "ast": 4,
                },
            ]
        }

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_player_game_log(203999, 2023)

            assert isinstance(result, list)
            assert len(result) == 2
            assert result[0]["pts"] == 30
            assert result[1]["pts"] == 28

    @patch("nba_predictor.data.balldontlie._get_api_key")
    def test_get_player_game_log_empty(self, mock_api_key):
        """Test game log with no games."""
        from nba_predictor.data import balldontlie

        mock_api_key.return_value = "test-api-key"

        expected_response = {"data": []}

        with patch("nba_predictor.data.balldontlie.requests.get") as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = expected_response
            mock_get.return_value = mock_response

            result = balldontlie.get_player_game_log(999999, 2023)

            assert isinstance(result, list)
            assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
