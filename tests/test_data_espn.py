import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json
import requests
import requests


class TestGetInjuries:
    """Tests for espn.get_injuries()"""

    @pytest.fixture
    def mock_response(self):
        # This is the format returned by the unofficial ESPN API scoreboard endpoint
        # which includes injury data within competitors
        return {
            "events": [
                {
                    "id": "401784512",
                    "date": "2024-04-02T19:30:00Z",
                    "competitions": [
                        {
                            "id": "401784512",
                            "type": {
                                "name": "Regular Season"
                            },
                            "competitors": [
                                {
                                    "id": "1610612738",
                                    "team": {
                                        "id": 1610612738,
                                        "location": "Boston",
                                        "name": "Celtics",
                                        "abbreviation": "BOS"
                                    },
                                    "injuries": [
                                        {
                                            "athlete": {
                                                "id": "12345",
                                                "fullName": "Jayson Tatum",
                                                "displayName": "Jayson Tatum"
                                            },
                                            "type": {
                                                "name": "DTD",
                                                "shortName": "DTD",
                                                "description": "Day-to-Day"
                                            },
                                            "status": "Out",
                                            "description": "Right knee injury",
                                            "returnDate": None,
                                            "startTime": None
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    def test_get_injuries_returns_dict_with_game_id(self, mock_response):
        """Test that get_injuries returns a dict with the game_id"""
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_injuries("401784512")
            
            assert isinstance(result, dict)
            assert result.get("gameId") == "401784512"

    def test_get_injuries_returns_injuries_list(self, mock_response):
        """Test that get_injuries returns injuries in the response"""
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_injuries("401784512")
            
            assert "injuries" in result
            assert isinstance(result["injuries"], list)
            assert len(result["injuries"]) == 1

    def test_get_injuries_handles_empty_injuries(self):
        """Test that get_injuries handles games with no injuries"""
        from nba_predictor.data import espn

        mock_response = {
            "events": [
                {
                    "id": "401784512",
                    "date": "2024-04-02T19:30:00Z",
                    "competitions": [
                        {
                            "id": "401784512",
                            "type": {"name": "Regular Season"},
                            "competitors": [
                                {
                                    "id": "1610612738",
                                    "team": {
                                        "id": 1610612738,
                                        "location": "Boston",
                                        "name": "Celtics",
                                        "abbreviation": "BOS"
                                    },
                                    "injuries": []
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_injuries("401784512")
            
            assert result.get("injuries") == []

    def test_get_injuries_caches_response(self, mock_response, tmp_path, monkeypatch):
        """Test that get_injuries caches the response"""
        import importlib
        from nba_predictor import config

        # Patch cache directory
        monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
        importlib.reload(config)
        importlib.reload(__import__("nba_predictor.data.espn", fromlist=["espn"]))

        from nba_predictor.data import espn

        # First call - should fetch
        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result1 = espn.get_injuries("401784512")
        
        # Second call - should use cache (no API call)
        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            result2 = espn.get_injuries("401784512")
        
        # Verify cache was used (mock_fetch should NOT be called on second run)
        assert mock_fetch.call_count == 0
        assert result1 == result2

    def test_get_injuries_retry_on_failure(self, mock_response):
        """Test that get_injuries retries with exponential backoff on failure"""
        from nba_predictor.data import espn

        # Mock with failures then success
        mock_response_first = {"error": "Rate limited"}
        
        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.side_effect = [
                requests.exceptions.RequestException("Network error"),
                mock_response         # Second call succeeds
            ]
            result = espn.get_injuries("401784512")
            
            assert result.get("gameId") == "401784512"
            assert mock_fetch.call_count == 2


class TestGetLineup:
    """Tests for espn.get_lineup()"""

    @pytest.fixture
    def mock_response(self):
        # This is the format returned by the unofficial ESPN API scoreboard endpoint
        # which includes lineup data within competitors
        return {
            "events": [
                {
                    "id": "401784512",
                    "date": "2024-04-02T19:30:00Z",
                    "competitions": [
                        {
                            "id": "401784512",
                            "type": {"name": "Regular Season"},
                            "competitors": [
                                {
                                    "id": "1610612738",
                                    "team": {
                                        "id": 1610612738,
                                        "location": "Boston",
                                        "name": "Celtics",
                                        "abbreviation": "BOS"
                                    },
                                    "lineup": [
                                        {
                                            "athlete": {
                                                "id": "12345",
                                                "fullName": "Jayson Tatum",
                                                "displayName": "Jayson Tatum"
                                            },
                                            "position": "SF",
                                            "status": "Active"
                                        },
                                        {
                                            "athlete": {
                                                "id": "67890",
                                                "fullName": "Jaylen Brown",
                                                "displayName": "Jaylen Brown"
                                            },
                                            "position": "SG",
                                            "status": "Active"
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }

    def test_get_lineup_returns_dict_with_game_id(self, mock_response):
        """Test that get_lineup returns a dict with the game_id"""
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_lineup("401784512")
            
            assert isinstance(result, dict)
            assert result.get("gameId") == "401784512"

    def test_get_lineup_returns_lineup_list(self, mock_response):
        """Test that get_lineup returns lineup in the response"""
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_lineup("401784512")
            
            assert "lineup" in result
            assert isinstance(result["lineup"], list)
            assert len(result["lineup"]) == 2

    def test_get_lineup_handles_empty_lineup(self):
        """Test that get_lineup handles games with no lineup info"""
        from nba_predictor.data import espn

        mock_response = {
            "events": [
                {
                    "id": "401784512",
                    "date": "2024-04-02T19:30:00Z",
                    "competitions": [
                        {
                            "id": "401784512",
                            "type": {"name": "Regular Season"},
                            "competitors": [
                                {
                                    "id": "1610612738",
                                    "team": {
                                        "id": 1610612738,
                                        "location": "Boston",
                                        "name": "Celtics",
                                        "abbreviation": "BOS"
                                    },
                                    "lineup": []
                                }
                            ]
                        }
                    ]
                }
            ]
        }

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_lineup("401784512")
            
            assert result.get("lineup") == []

    def test_get_lineup_caches_response(self, mock_response, tmp_path, monkeypatch):
        """Test that get_lineup caches the response"""
        import importlib
        from nba_predictor import config

        # Patch cache directory
        monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
        importlib.reload(config)
        importlib.reload(__import__("nba_predictor.data.espn", fromlist=["espn"]))

        from nba_predictor.data import espn

        # First call - should fetch
        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result1 = espn.get_lineup("401784512")
        
        # Second call - should use cache
        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            result2 = espn.get_lineup("401784512")
        
        # Verify cache was used
        assert mock_fetch.call_count == 0
        assert result1 == result2


class TestGetTeamStatus:
    """Tests for espn.get_team_status()"""

    @pytest.fixture
    def mock_response(self):
        return {
            "team": {
                "id": 1610612738,
                "location": "Boston",
                "name": "Celtics",
                "abbreviation": "BOS"
            },
            "events": [
                {
                    "id": "401784500",
                    "date": "2024-04-01T19:00:00Z",
                    "status": {
                        "type": {
                            "name": "STATUS_FINAL"
                        }
                    }
                },
                {
                    "id": "401784512",
                    "date": "2024-04-04T19:30:00Z",
                    "status": {
                        "type": {
                            "name": "STATUS_SCHEDULED"
                        }
                    }
                }
            ]
        }

    def test_get_team_status_returns_dict_with_team_id(self, mock_response):
        """Test that get_team_status returns a dict with the team_id"""
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_team_status(1610612738)
            
            assert isinstance(result, dict)
            assert result.get("teamId") == 1610612738

    def test_get_team_status_returns_rest_days(self, mock_response):
        """Test that get_team_status returns rest days info"""
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_team_status(1610612738)
            
            assert "restDays" in result

    def test_get_team_status_returns_back_to_back_info(self, mock_response):
        """Test that get_team_status returns back-to-back info"""
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_team_status(1610612738)
            
            assert "backToBack" in result

    def test_get_team_status_handles_zero_rest_days(self):
        """Test that get_team_status handles zero rest days (back-to-back)"""
        from nba_predictor.data import espn

        mock_response = {
            "team": {
                "id": 1610612738,
                "location": "Boston",
                "name": "Celtics",
                "abbreviation": "BOS"
            },
            "events": [
                {
                    "id": "401784500",
                    "date": "2024-04-01T19:00:00Z",
                    "status": {
                        "type": {
                            "name": "STATUS_FINAL"
                        }
                    }
                },
                {
                    "id": "401784512",
                    "date": "2024-04-02T19:30:00Z",
                    "status": {
                        "type": {
                            "name": "STATUS_SCHEDULED"
                        }
                    }
                }
            ]
        }

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result = espn.get_team_status(1610612738)
            
            assert "restDays" in result
            assert "backToBack" in result

    def test_get_team_status_caches_response(self, mock_response, tmp_path, monkeypatch):
        """Test that get_team_status caches the response"""
        import importlib
        from nba_predictor import config

        # Patch cache directory
        monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
        importlib.reload(config)
        importlib.reload(__import__("nba_predictor.data.espn", fromlist=["espn"]))

        from nba_predictor.data import espn

        # First call - should fetch
        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.return_value = mock_response
            result1 = espn.get_team_status(1610612738)
        
        # Second call - should use cache
        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            result2 = espn.get_team_status(1610612738)
        
        # Verify cache was used
        assert mock_fetch.call_count == 0
        assert result1 == result2


class TestErrorHandling:
    """Tests for error handling in espn module"""

    def test_get_injuries_handles_network_error(self):
        """Test that get_injuries raises appropriate error on network failure"""
        import requests
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.side_effect = requests.exceptions.RequestException("Network error")
            
            with pytest.raises(requests.exceptions.RequestException):
                espn.get_injuries("401784512")

    def test_get_lineup_handles_network_error(self):
        """Test that get_lineup raises appropriate error on network failure"""
        import requests
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.side_effect = requests.exceptions.RequestException("Network error")
            
            with pytest.raises(requests.exceptions.RequestException):
                espn.get_lineup("401784512")

    def test_get_team_status_handles_network_error(self):
        """Test that get_team_status raises appropriate error on network failure"""
        import requests
        from nba_predictor.data import espn

        with patch("nba_predictor.data.espn._fetch_espn_data") as mock_fetch:
            mock_fetch.side_effect = requests.exceptions.RequestException("Network error")
            
            with pytest.raises(requests.exceptions.RequestException):
                espn.get_team_status(1610612738)
