"""GET /snapshot-meta: when the numbers this API serves were last written.

NBA is the odd one out of the five sport APIs: its routes never read
public_snapshot.json. They read the tracking DB (config.TRACKING_DB_PATH) and
the hub caches (config.DATA_DIR / "cache" / "hub" / *.json, via
load_hub_cache), so there is no generated_at to report verbatim. The honest
equivalent is the newest mtime across exactly those files. Reporting the
snapshot file's timestamp instead would describe numbers the visitor is not
looking at -- that was ruled out deliberately, and the next session must not
"fix" this endpoint to read the snapshot.
"""

import json
import os
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from nba_predictor import config
from nba_predictor.api.app import app


def _point_config_at(tmp_path, monkeypatch, *, with_files):
    """Redirect the files the routes actually read into an empty tmp dir.

    TRACKING_DB_PATH is env-overridable (config reads TRACKING_DB_PATH from
    the environment at import), so it can legally point at a path that does
    not exist locally -- the endpoint's exists() guard must cover that, and
    the missing-files test below pins it.
    """
    monkeypatch.setattr(config, "TRACKING_DB_PATH", tmp_path / "tracking.db")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    if with_files:
        hub_dir = tmp_path / "cache" / "hub"
        hub_dir.mkdir(parents=True)
        # Distinct mtimes so the test can tell which file won: the hub cache
        # is newer than the DB, so the newest mtime must come from it.
        db_path = tmp_path / "tracking.db"
        db_path.write_bytes(b"db")
        hub_file = hub_dir / "teams.json"
        hub_file.write_text(json.dumps([{"abbreviation": "BOS"}]))
        os.utime(db_path, (1_700_000_000, 1_700_000_000))
        os.utime(hub_file, (1_700_000_100, 1_700_000_100))
        return hub_file
    return None


def test_snapshot_meta_reports_newest_mtime(tmp_path, monkeypatch):
    newest = _point_config_at(tmp_path, monkeypatch, with_files=True)
    client = TestClient(app)
    response = client.get("/snapshot-meta")
    assert response.status_code == 200
    body = response.json()
    assert body["source"] == "public_snapshot"
    # The timestamp is derived from the filesystem, so pin it to the file
    # this checkout's real data cannot influence: the newest tmp file.
    expected = datetime.fromtimestamp(
        newest.stat().st_mtime, tz=timezone.utc
    ).isoformat()
    assert body["generated_at"] == expected
    # ... and it must be a valid ISO-8601 UTC string, not just any string.
    parsed = datetime.fromisoformat(body["generated_at"])
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_snapshot_meta_missing_db_still_reports_hub_cache(tmp_path, monkeypatch):
    """The exists() guard covers a TRACKING_DB_PATH that points nowhere --
    e.g. an env override with no local file -- as long as a hub cache exists.
    """
    hub_dir = tmp_path / "cache" / "hub"
    hub_dir.mkdir(parents=True)
    hub_file = hub_dir / "standings.json"
    hub_file.write_text(json.dumps([]))
    os.utime(hub_file, (1_700_000_200, 1_700_000_200))
    monkeypatch.setattr(
        config, "TRACKING_DB_PATH", tmp_path / "does-not-exist.db"
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    client = TestClient(app)
    body = client.get("/snapshot-meta").json()
    assert body["source"] == "public_snapshot"
    assert body["generated_at"] == datetime.fromtimestamp(
        hub_file.stat().st_mtime, tz=timezone.utc
    ).isoformat()


def test_snapshot_meta_schedule_newest_wins(tmp_path, monkeypatch):
    """The schedule cache (games.json) feeds /games/week -- the endpoint the
    hub's NBA teaser calls -- so when it is the newest file on disk its mtime
    must win. Fails against a path list that only watches the DB and the hub
    caches, which would report the older hub mtime instead."""
    _point_config_at(tmp_path, monkeypatch, with_files=True)
    sched_dir = tmp_path / "cache" / "schedule"
    sched_dir.mkdir(parents=True)
    sched_file = sched_dir / "games.json"
    sched_file.write_text(json.dumps([]))
    os.utime(sched_file, (1_700_000_500, 1_700_000_500))
    client = TestClient(app)
    body = client.get("/snapshot-meta").json()
    assert body["source"] == "public_snapshot"
    assert body["generated_at"] == datetime.fromtimestamp(
        sched_file.stat().st_mtime, tz=timezone.utc
    ).isoformat()


def test_snapshot_meta_nothing_written_yet_is_live_not_an_error(
    tmp_path, monkeypatch
):
    """Every path missing -- a public deploy before its first refresh -- is
    the real "live" state, not a failure, so this must not raise."""
    _point_config_at(tmp_path, monkeypatch, with_files=False)
    client = TestClient(app)
    response = client.get("/snapshot-meta")
    assert response.status_code == 200
    assert response.json() == {"generated_at": None, "source": "live"}


def test_snapshot_meta_reachable_at_real_route_path(tmp_path, monkeypatch):
    """The router is included with no prefix (app.include_router(router) in
    api/app.py), so the real path is /snapshot-meta, not /api/snapshot-meta.
    """
    _point_config_at(tmp_path, monkeypatch, with_files=False)
    client = TestClient(app)
    response = client.get("/snapshot-meta")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"generated_at", "source"}
