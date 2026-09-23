# tests/test_tracking_store.py
import sqlite3

import pytest


def test_insert_and_get_player_outcomes_roundtrip(tmp_path):
    from nba_predictor.tracking import store
    import pytest

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    row_id = store.insert_player_outcome(
        db_path, game_id="g1", player_id="203999", stat="points",
        actual_value=24.0, recorded_at="2026-11-01T22:00:00",
    )
    assert row_id == 1

    rows = store.get_player_outcomes_for_game(db_path, "g1")
    assert len(rows) == 1
    assert rows[0]["player_id"] == "203999"
    assert rows[0]["stat"] == "points"
    assert rows[0]["actual_value"] == pytest.approx(24.0)


def test_get_player_outcomes_for_game_returns_empty_for_unknown_game(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    assert store.get_player_outcomes_for_game(db_path, "does-not-exist") == []


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


def test_insert_market_prediction_point_defaults_to_none(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="BOS", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=-110,
        created_at="2026-11-01T12:00:00",
    )

    rows = store.get_market_predictions_for_game(db_path, "g1")
    assert rows[0]["point"] is None


def test_insert_market_prediction_stores_point_line_value(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    store.insert_market_prediction(
        db_path, game_id="g1", market="spread", selection="BOS", model_probability=0.55,
        market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=-110,
        point=-4.5, created_at="2026-11-01T12:00:00",
    )

    rows = store.get_market_predictions_for_game(db_path, "g1")
    assert rows[0]["point"] == pytest.approx(-4.5)


def test_position_column_migration_preserves_rows(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    # Build an old-schema DB by hand (no position column).
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE player_prediction_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT NOT NULL,
            player_id TEXT NOT NULL, stat TEXT NOT NULL,
            predicted_value REAL NOT NULL, created_at TEXT NOT NULL)"""
    )
    conn.execute(
        "INSERT INTO player_prediction_snapshots (game_id, player_id, stat, predicted_value, created_at)"
        " VALUES ('g1', 'p1', 'points', 20.5, '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    store.init_db(db_path)
    with store.get_connection(db_path) as c:
        cols = {r[1] for r in c.execute("PRAGMA table_info(player_prediction_snapshots)")}
        assert "position" in cols
        rows = c.execute(
            "SELECT game_id, player_id, predicted_value, position FROM player_prediction_snapshots"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["game_id"] == "g1" and rows[0]["position"] is None


def test_insert_player_prediction_stores_position(tmp_path):
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="p1", stat="points",
        predicted_value=20.5, created_at="2026-01-01T00:00:00+00:00", position="G",
    )
    store.insert_player_prediction(
        db_path, game_id="g1", player_id="p2", stat="points",
        predicted_value=10.0, created_at="2026-01-01T00:00:00+00:00",
    )
    rows = store.get_player_predictions_for_game(db_path, "g1")
    by_player = {r["player_id"]: r["position"] for r in rows}
    assert by_player == {"p1": "G", "p2": None}
