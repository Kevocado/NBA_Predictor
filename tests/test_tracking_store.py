# tests/test_tracking_store.py
import sqlite3

import pytest


def test_init_db_creates_all_six_tables(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    table_names = {row[0] for row in rows}

    expected = {
        "predictions",
        "game_market_predictions",
        "game_forecast_snapshots",
        "odds_timing_snapshots",
        "player_prediction_snapshots",
        "game_player_outcomes",
    }
    assert expected.issubset(table_names)


def test_init_db_is_idempotent(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.init_db(db_path)  # must not raise


def test_insert_and_get_predictions_roundtrip(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    row_id = store.insert_prediction(
        db_path,
        game_id="0022500123",
        created_at="2026-11-01T12:00:00",
        model_version="v1",
        home_win_prob=0.62,
        predicted_margin=3.5,
        predicted_total=224.5,
    )
    assert row_id == 1

    rows = store.get_predictions_for_game(db_path, "0022500123")
    assert len(rows) == 1
    assert rows[0]["game_id"] == "0022500123"
    assert rows[0]["home_win_prob"] == pytest.approx(0.62)
    assert rows[0]["model_version"] == "v1"


def test_get_predictions_for_game_returns_empty_for_unknown_game(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    rows = store.get_predictions_for_game(db_path, "does-not-exist")
    assert rows == []


def test_get_all_predictions_returns_latest_per_game(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-11-01T08:00:00", model_version="v1",
        home_win_prob=0.55, predicted_margin=1.0, predicted_total=220.0,
    )
    store.insert_prediction(
        db_path, game_id="g1", created_at="2026-11-01T18:00:00", model_version="v1",
        home_win_prob=0.60, predicted_margin=2.0, predicted_total=222.0,
    )
    store.insert_prediction(
        db_path, game_id="g2", created_at="2026-11-01T09:00:00", model_version="v1",
        home_win_prob=0.40, predicted_margin=-1.0, predicted_total=210.0,
    )

    rows = store.get_all_predictions(db_path)
    by_game = {row["game_id"]: row for row in rows}

    assert len(rows) == 2
    assert by_game["g1"]["home_win_prob"] == pytest.approx(0.60)
    assert by_game["g2"]["home_win_prob"] == pytest.approx(0.40)


def test_get_all_predictions_empty_db_returns_empty_list(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert store.get_all_predictions(db_path) == []
