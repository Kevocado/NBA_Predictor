import sqlite3

import pytest

from nba_predictor.tracking import store


def _rows(n):
    return [dict(game_id="g", player_id=f"p{i}", stat="points", predicted_value=float(i), created_at="2026-10-21T00:00:00+00:00", position="G") for i in range(n)]


def test_bulk_insert_writes_every_row_in_one_transaction(tmp_path):
    db = tmp_path / "t.db"
    store.init_db(db)
    assert store.insert_player_predictions(db, _rows(500)) == 500
    got = store.get_player_predictions_for_game(db, "g")
    assert len(got) == 500 and got[7]["predicted_value"] == 7.0


def test_a_transient_disk_error_is_retried(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    store.init_db(db)
    real = store._connect
    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("disk I/O error")
        return real(path)

    monkeypatch.setattr(store, "_connect", flaky)
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    assert store.insert_player_predictions(db, _rows(3)) == 3


def test_a_persistent_error_is_raised_not_swallowed(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    store.init_db(db)
    monkeypatch.setattr(store, "_connect", lambda path: (_ for _ in ()).throw(sqlite3.OperationalError("disk I/O error")))
    monkeypatch.setattr(store.time, "sleep", lambda s: None)
    with pytest.raises(sqlite3.OperationalError):
        store.insert_player_predictions(db, _rows(3))
