from unittest.mock import patch

import pytest


def _real_event(home_name="Boston Celtics", away_name="Miami Heat", event_date="2026-11-01"):
    """Shape captured live from GET /v0/competitions/{key}/events."""
    return {
        "key": "evt-1",
        "startTime": f"{event_date}T23:30:00.000Z",
        "homeParticipantKey": "home-key",
        "participants": [
            {"key": "home-key", "name": home_name},
            {"key": "away-key", "name": away_name},
        ],
    }


def test_match_schedule_to_sportsbook_events_matches_by_team_and_date():
    from nba_predictor.pipeline.refresh_odds import match_schedule_to_sportsbook_events

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]
    events = [_real_event()]

    mapping = match_schedule_to_sportsbook_events(schedule, events)

    assert mapping == {"g1": "evt-1"}


def test_match_schedule_to_sportsbook_events_skips_unmatched_games():
    from nba_predictor.pipeline.refresh_odds import match_schedule_to_sportsbook_events

    schedule = [{"game_id": "g1", "game_date": "2026-11-05", "home_team": "BOS", "away_team": "MIA"}]
    events = [_real_event(event_date="2026-11-01")]

    assert match_schedule_to_sportsbook_events(schedule, events) == {}


def test_match_schedule_to_sportsbook_events_skips_malformed_participants():
    from nba_predictor.pipeline.refresh_odds import match_schedule_to_sportsbook_events

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}]
    events = [{"key": "evt-1", "startTime": "2026-11-01T00:00:00.000Z", "homeParticipantKey": "x", "participants": []}]

    assert match_schedule_to_sportsbook_events(schedule, events) == {}


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_stores_devigged_h2h_rows(mock_events, mock_odds, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-30T00:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=3.0, predicted_total=220.0,
    )

    mock_events.return_value = [_real_event()]
    mock_odds.return_value = [
        {"market": "h2h", "selection": "BOS", "bookmaker": "DRAFT_KINGS", "american_odds": -140, "point": None},
        {"market": "h2h", "selection": "MIA", "bookmaker": "DRAFT_KINGS", "american_odds": 120, "point": None},
    ]

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    stored = refresh_market_predictions(schedule, db_path)

    assert stored == 2
    rows = store.get_market_predictions_for_game(db_path, "g1")
    by_selection = {r["selection"]: r for r in rows}
    assert by_selection["BOS"]["model_probability"] == 0.62
    assert by_selection["MIA"]["model_probability"] == pytest.approx(0.38)
    assert by_selection["BOS"]["market_probability"] is not None
    assert by_selection["BOS"]["edge"] is not None


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_skips_completed_games(mock_events, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    schedule = [{"game_id": "g1", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA", "completed": True}]

    stored = refresh_market_predictions(schedule, db_path)

    assert stored == 0
    mock_events.assert_not_called()


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_returns_zero_when_no_events_match(mock_events, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    mock_events.return_value = []
    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    assert refresh_market_predictions(schedule, db_path) == 0


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_skips_games_with_no_stored_prediction(mock_events, mock_odds, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    mock_events.return_value = [_real_event()]
    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    stored = refresh_market_predictions(schedule, db_path)

    assert stored == 0
    mock_odds.assert_not_called()


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_stores_devigged_spread_rows(mock_events, mock_odds, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-30T00:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=5.0, predicted_total=220.0,
    )

    mock_events.return_value = [_real_event()]
    mock_odds.return_value = [
        {"market": "spread", "selection": "BOS", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": -4.5},
        {"market": "spread", "selection": "MIA", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 4.5},
    ]

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    stored = refresh_market_predictions(schedule, db_path, margin_std=10.0)

    assert stored == 2
    rows = store.get_market_predictions_for_game(db_path, "g1")
    by_selection = {r["selection"]: r for r in rows}
    assert by_selection["BOS"]["market"] == "spread"
    assert by_selection["BOS"]["point"] == pytest.approx(-4.5)
    assert by_selection["BOS"]["model_probability"] > 0.5
    assert by_selection["MIA"]["model_probability"] == pytest.approx(1 - by_selection["BOS"]["model_probability"])


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_stores_devigged_total_rows(mock_events, mock_odds, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-30T00:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=5.0, predicted_total=230.0,
    )

    mock_events.return_value = [_real_event()]
    mock_odds.return_value = [
        {"market": "total", "selection": "over", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 220.5},
        {"market": "total", "selection": "under", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 220.5},
    ]

    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    stored = refresh_market_predictions(schedule, db_path, total_std=10.0)

    assert stored == 2
    rows = store.get_market_predictions_for_game(db_path, "g1")
    by_selection = {r["selection"]: r for r in rows}
    assert by_selection["over"]["point"] == pytest.approx(220.5)
    assert by_selection["over"]["model_probability"] > 0.5
    assert by_selection["under"]["model_probability"] == pytest.approx(1 - by_selection["over"]["model_probability"])


@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.get_odds")
@patch("nba_predictor.pipeline.refresh_odds.sportsbook_api.fetch_nba_events_raw")
def test_refresh_market_predictions_skips_lopsided_market_groups(mock_events, mock_odds, tmp_path):
    from nba_predictor.pipeline.refresh_odds import refresh_market_predictions
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-10-30T00:00:00", model_version="v1",
        home_win_prob=0.62, predicted_margin=5.0, predicted_total=230.0,
    )

    mock_events.return_value = [_real_event()]
    mock_odds.return_value = [
        {"market": "total", "selection": "over", "bookmaker": "DRAFT_KINGS", "american_odds": -110, "point": 220.5},
    ]
    schedule = [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA", "completed": False}]

    assert refresh_market_predictions(schedule, db_path) == 0
