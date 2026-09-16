import json


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_generate_snapshot_bundles_all_sources(tmp_path):
    from nba_predictor.public_snapshot import generate_snapshot
    from nba_predictor.tracking import store

    schedule_path = tmp_path / "schedule" / "games.json"
    _write_json(schedule_path, [{"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}])

    hub_dir = tmp_path / "hub"
    _write_json(hub_dir / "teams.json", [{"abbreviation": "BOS"}])
    _write_json(hub_dir / "players.json", [])
    _write_json(hub_dir / "rankings.json", [])
    _write_json(hub_dir / "standings.json", [])

    manifest_path = tmp_path / "models" / "manifest.json"
    _write_json(manifest_path, {"model_version": "v1", "trained_at": "t", "models": [], "metrics": {}})

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    store.insert_market_prediction(
        db_path, game_id="g1", market="h2h", selection="home", model_probability=0.6,
        market_probability=0.5, edge=0.1, bookmaker="DK", american_odds=-130, created_at="2026-11-01T00:00:00",
    )

    snapshot = generate_snapshot(schedule_path, hub_dir, manifest_path, db_path)

    assert snapshot["schedule"][0]["game_id"] == "g1"
    assert snapshot["hub"]["teams"][0]["abbreviation"] == "BOS"
    assert snapshot["hub"]["players"] == []
    assert snapshot["manifest"]["model_version"] == "v1"
    assert snapshot["track_record"][0]["market"] == "h2h"
    assert "generated_at" in snapshot


def test_generate_snapshot_manifest_none_when_missing(tmp_path):
    from nba_predictor.public_snapshot import generate_snapshot
    from nba_predictor.tracking import store

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)

    snapshot = generate_snapshot(
        tmp_path / "schedule" / "games.json", tmp_path / "hub", tmp_path / "models" / "manifest.json", db_path
    )
    assert snapshot["manifest"] is None


def test_write_snapshot_creates_valid_json(tmp_path):
    from nba_predictor.public_snapshot import write_snapshot

    output_path = tmp_path / "public_snapshot.json"
    write_snapshot({"generated_at": "2026-11-01T00:00:00", "schedule": []}, output_path)

    with open(output_path) as f:
        loaded = json.load(f)
    assert loaded["schedule"] == []


def test_main_writes_to_configured_output_path(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.public_snapshot import main
    from nba_predictor.tracking import store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    store.init_db(tmp_path / "tracking.db")

    main()

    assert (tmp_path / "public_snapshot.json").exists()


def test_main_initializes_tracking_db_when_it_does_not_exist_yet(tmp_path, monkeypatch):
    from nba_predictor import config
    from nba_predictor.public_snapshot import main

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")

    main()  # must not raise even though tracking.db has never been created

    assert (tmp_path / "tracking.db").exists()
    assert (tmp_path / "public_snapshot.json").exists()
