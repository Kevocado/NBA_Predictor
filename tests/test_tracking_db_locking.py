"""SQLite's own locking must be ON unless something explicitly turns it off.

The tracking DB used to open every connection with `?nolock=1`. That was written
for the Azure Files (SMB) mount, where fcntl locks are not honoured, and it was
justified by "the container app is pinned to exactly one replica, making this
process the sole writer". That sentence was false: the daily `nba-rescore`
systemd unit runs `docker exec ... python -m nba_predictor.pipeline.refresh_full`,
a SECOND process in the same container writing the same file. `_DB_LOCK` is a
`threading.Lock`, so it serialises threads and nothing else.

On 2026-10-26 that produced a real outage: the rescore died at
`sqlite3.OperationalError: disk I/O error` on a commit, and the file was left
malformed -- `sqlite3.DatabaseError: database disk image is malformed` on every
read, so the live games page 500'd.

The DB now lives on a local ext4 volume (compose binds `./volumes/nba-tracking`),
where locking works. It stays available as an opt-in for SMB, because that is
the only place it is needed.
"""
from pathlib import Path

import sqlite3

import pytest

from nba_predictor.tracking import store


@pytest.fixture
def captured_uri(monkeypatch):
    """Capture the URI `_connect` would open, without touching a real file."""
    seen: dict[str, str] = {}
    real_connect = sqlite3.connect

    def fake_connect(database, **kwargs):
        seen["uri"] = str(database)
        return real_connect(":memory:")

    monkeypatch.setattr(store.sqlite3, "connect", fake_connect)
    return seen


def test_locking_is_not_disabled_by_default(captured_uri, monkeypatch):
    monkeypatch.delenv(store.NOLOCK_ENV, raising=False)
    store._connect(Path("/tmp/tracking.db"))

    assert "nolock=1" not in captured_uri["uri"], (
        "the connection disables SQLite locking by default; a second process writing "
        "the same DB is how the file was corrupted"
    )


def test_locking_can_still_be_disabled_for_smb(captured_uri, monkeypatch):
    """The escape hatch exists for Azure Files, where fcntl locks do not work."""
    monkeypatch.setenv(store.NOLOCK_ENV, "1")
    store._connect(Path("/tmp/tracking.db"))

    assert "nolock=1" in captured_uri["uri"]


def test_the_opt_in_is_case_and_space_insensitive(captured_uri, monkeypatch):
    monkeypatch.setenv(store.NOLOCK_ENV, " TRUE ")
    store._connect(Path("/tmp/tracking.db"))

    assert "nolock=1" in captured_uri["uri"]


def test_a_false_value_does_not_disable_locking(captured_uri, monkeypatch):
    monkeypatch.setenv(store.NOLOCK_ENV, "0")
    store._connect(Path("/tmp/tracking.db"))

    assert "nolock=1" not in captured_uri["uri"]
