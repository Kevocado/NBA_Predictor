from unittest.mock import patch

import pytest


@pytest.fixture
def clear_cache(tmp_path, monkeypatch):
    from nba_predictor.data import sportsbook_api

    monkeypatch.setattr(sportsbook_api, "CACHE_DIR", tmp_path / "sportsbook")
    monkeypatch.setattr(sportsbook_api.config, "SPORTSBOOK_API_KEY", "test-key")
    return sportsbook_api.CACHE_DIR


def test_decimal_to_american_favorite():
    from nba_predictor.data.sportsbook_api import decimal_to_american

    assert decimal_to_american(1.91) == -110


def test_decimal_to_american_underdog():
    from nba_predictor.data.sportsbook_api import decimal_to_american

    assert decimal_to_american(9.5) == 850


def test_get_api_key_raises_when_missing(monkeypatch):
    from nba_predictor import config
    from nba_predictor.data.sportsbook_api import SportsbookAPIKeyMissing, _get_api_key

    monkeypatch.setattr(config, "SPORTSBOOK_API_KEY", None)

    with pytest.raises(SportsbookAPIKeyMissing):
        _get_api_key()


def _real_spread_event():
    """Shape captured live from GET /v0/events?eventKeys= on a real WNBA game."""
    return {
        "key": "annT-Ajrx-meld",
        "markets": [
            {
                "type": "MONEYLINE",
                "participantKey": None,
                "outcomes": {
                    "DRAFT_KINGS": [
                        {
                            "modifier": 0, "payout": 1.5, "type": "WIN",
                            "participantKey": "away-key",
                            "participant": {"name": "Boston Celtics", "shortName": "BOS"},
                        },
                        {
                            "modifier": 0, "payout": 2.7, "type": "WIN",
                            "participantKey": "home-key",
                            "participant": {"name": "Miami Heat", "shortName": "MIA"},
                        },
                    ]
                },
            },
            {
                "type": "POINT_SPREAD",
                "participantKey": None,
                "outcomes": {
                    "DRAFT_KINGS": [
                        {
                            "modifier": 15.5, "payout": 1.91, "type": "WIN",
                            "participant": {"name": "Boston Celtics", "shortName": "BOS"},
                        },
                        {
                            "modifier": -15.5, "payout": 1.91, "type": "WIN",
                            "participant": {"name": "Miami Heat", "shortName": "MIA"},
                        },
                    ]
                },
            },
            {
                "type": "POINT_TOTAL",
                "participantKey": None,
                "outcomes": {
                    "DRAFT_KINGS": [
                        {"modifier": 168.5, "payout": 1.89, "type": "OVER", "participant": None},
                        {"modifier": 168.5, "payout": 1.92, "type": "UNDER", "participant": None},
                    ]
                },
            },
        ],
    }


@patch("nba_predictor.data.sportsbook_api._get")
def test_fetch_nba_events_raw_returns_events(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import fetch_nba_events_raw

    mock_get.return_value = {"events": [{"key": "e1", "name": "BOS @ MIA"}]}

    events = fetch_nba_events_raw()

    assert events == [{"key": "e1", "name": "BOS @ MIA"}]


@patch("nba_predictor.data.sportsbook_api._get")
def test_fetch_nba_events_raw_caches_across_calls(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import fetch_nba_events_raw

    mock_get.return_value = {"events": []}

    fetch_nba_events_raw()
    fetch_nba_events_raw()

    assert mock_get.call_count == 1


@patch("nba_predictor.data.sportsbook_api._get")
def test_fetch_event_odds_raw_unwraps_nested_groups(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import fetch_event_odds_raw

    mock_get.return_value = {"events": [[_real_spread_event()]]}

    event = fetch_event_odds_raw("annT-Ajrx-meld")

    assert event["key"] == "annT-Ajrx-meld"


@patch("nba_predictor.data.sportsbook_api._get")
def test_fetch_event_odds_raw_returns_none_for_empty_response(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import fetch_event_odds_raw

    mock_get.return_value = {"events": []}

    assert fetch_event_odds_raw("missing") is None


@patch("nba_predictor.data.sportsbook_api._get")
def test_get_odds_maps_team_markets_with_selection_and_american_odds(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import get_odds

    mock_get.return_value = {"events": [[_real_spread_event()]]}

    rows = get_odds("annT-Ajrx-meld")
    by_market = {}
    for row in rows:
        by_market.setdefault(row["market"], []).append(row)

    assert len(by_market["h2h"]) == 2
    assert len(by_market["spread"]) == 2
    assert len(by_market["total"]) == 2

    spread_bos = next(r for r in by_market["spread"] if r["selection"] == "BOS")
    assert spread_bos["point"] == 15.5
    assert spread_bos["american_odds"] == -110

    total_over = next(r for r in by_market["total"] if r["selection"] == "over")
    assert total_over["point"] == 168.5


@patch("nba_predictor.data.sportsbook_api._get")
def test_get_odds_returns_empty_list_for_missing_event(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import get_odds

    mock_get.return_value = {"events": []}

    assert get_odds("missing") == []


@patch("nba_predictor.data.sportsbook_api._get")
def test_get_player_props_returns_empty_when_no_player_markets_posted(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import get_player_props

    mock_get.return_value = {"events": [[_real_spread_event()]]}

    assert get_player_props("annT-Ajrx-meld") == []


@patch("nba_predictor.data.sportsbook_api._get")
def test_get_player_props_extracts_markets_with_participant_key(mock_get, clear_cache):
    from nba_predictor.data.sportsbook_api import get_player_props

    event = {
        "key": "e1",
        "markets": [
            {
                "type": "PLAYER_POINTS",
                "participantKey": "player-key",
                "participant": {"name": "Jayson Tatum"},
                "outcomes": {
                    "DRAFT_KINGS": [
                        {"modifier": 27.5, "payout": 1.91, "type": "OVER"},
                    ]
                },
            }
        ],
    }
    mock_get.return_value = {"events": [[event]]}

    props = get_player_props("e1")

    assert len(props) == 1
    assert props[0]["player"] == "Jayson Tatum"
    assert props[0]["point"] == 27.5
