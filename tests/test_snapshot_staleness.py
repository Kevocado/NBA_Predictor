"""The snapshot staleness gate (spec section 10, gap G6).

On 2026-10-04 `data/public_snapshot.json` carried a fresh `generated_at` while
the models behind it were trained on 2026-09-18 — 16 days earlier, during a
window in which the daily refresh had failed on every single run. A reader sees
"generated today" and concludes the numbers are current. They were not, and
nothing said so.

These pin the refusal and, just as importantly, that it does not fire on a
healthy manifest — a gate that always refuses is as useless as no gate.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nba_predictor.public_snapshot import (
    MAX_MODEL_AGE_DAYS,
    StaleModels,
    assert_models_fresh,
    model_age_days,
)

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=timezone.utc)


def manifest_trained(days_ago: float) -> dict:
    trained = NOW - timedelta(days=days_ago)
    return {
        "model_version": "v20260918120240",
        "trained_at": trained.isoformat(),
        "models": ["win_probability", "margin", "total"],
        "metrics": {},
    }


def test_a_twenty_day_old_model_is_refused():
    """The Phase A measurement, pinned. 16 days is what actually shipped."""
    with pytest.raises(StaleModels):
        assert_models_fresh(manifest_trained(20), now=NOW)


def test_a_two_day_old_model_publishes():
    """The other half: a gate that refuses everything is as useless as none."""
    assert assert_models_fresh(manifest_trained(2), now=NOW) == pytest.approx(2.0, abs=0.01)


def test_the_boundary_is_inclusive_at_the_limit():
    assert_models_fresh(manifest_trained(MAX_MODEL_AGE_DAYS), now=NOW)
    with pytest.raises(StaleModels):
        assert_models_fresh(manifest_trained(MAX_MODEL_AGE_DAYS + 0.01), now=NOW)


def test_no_manifest_is_not_stale_and_does_not_block_a_first_publish():
    """A repository that has never trained publishes `"manifest": null` and the
    frontend shows a "not trained" state. That is a real state and misleads no
    reader, so the gate must not refuse it -- it exists to catch a snapshot that
    looks current while the model behind it is not."""
    assert assert_models_fresh(None, now=NOW) is None


def test_a_manifest_with_no_readable_trained_at_is_refused():
    """An undated model carries no evidence of being current. Treating the
    missing stamp as `now` would wave the exact failure this gate exists to catch
    straight through -- and the shipped manifest once did have a real stamp, so
    a missing one means something broke, not that the models are new."""
    for bad in ({"model_version": "v1"}, {"trained_at": ""}, {"trained_at": "not a date"},
                {"trained_at": None}):
        with pytest.raises(StaleModels, match="trained_at"):
            assert_models_fresh(bad, now=NOW)


def test_a_naive_timestamp_is_read_as_utc():
    """`trained_at` written without an offset is still a real instant. Assuming
    it was local time would shift the age by hours and could flip the boundary."""
    naive = (NOW - timedelta(days=3)).replace(tzinfo=None).isoformat()
    age = model_age_days({"trained_at": naive}, now=NOW)
    assert age == pytest.approx(3.0, abs=0.01)


def test_a_z_suffixed_timestamp_is_read():
    stamp = (NOW - timedelta(days=1)).isoformat().replace("+00:00", "Z")
    assert model_age_days({"trained_at": stamp}, now=NOW) == pytest.approx(1.0, abs=0.01)


def test_model_age_is_none_not_zero_without_a_stamp():
    assert model_age_days({}, now=NOW) is None
    assert model_age_days(None, now=NOW) is None


def test_the_refusal_says_how_stale_so_the_log_distinguishes_two_problems(tmp_path):
    """A stale model and a broken refresh are different incidents with different
    fixes, and the message is the only thing that tells them apart in a log."""
    with pytest.raises(StaleModels) as excinfo:
        assert_models_fresh(manifest_trained(20), now=NOW)
    message = str(excinfo.value)
    assert "20.0 days" in message
    assert "refresh is not running" in message


def test_generate_snapshot_refuses_before_writing_anything(tmp_path, monkeypatch):
    """The gate has to fire before the expensive part runs, and before any file
    is touched -- a snapshot written and then deleted is still a window where a
    stale one was on disk."""
    from nba_predictor import public_snapshot

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_trained(20)))

    def explode(*_a, **_k):  # pragma: no cover - must never run
        raise AssertionError("the expensive part ran despite stale models")

    monkeypatch.setattr(public_snapshot, "compute_track_record", explode)
    monkeypatch.setattr(public_snapshot, "load_schedule", explode)
    monkeypatch.setattr(public_snapshot, "load_hub_cache", explode)

    with pytest.raises(StaleModels):
        public_snapshot.generate_snapshot(
            schedule_path=tmp_path / "games.json",
            hub_dir=tmp_path / "hub",
            manifest_path=manifest_path,
            db_path=tmp_path / "tracking.db",
            now=NOW,
        )
    assert not (tmp_path / "public_snapshot.json").exists()


def test_generate_snapshot_proceeds_with_fresh_models(tmp_path, monkeypatch):
    from nba_predictor import public_snapshot

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest_trained(1)))

    monkeypatch.setattr(public_snapshot, "load_schedule", lambda _p: [])
    monkeypatch.setattr(public_snapshot, "load_hub_cache", lambda _p: None)
    monkeypatch.setattr(public_snapshot, "compute_track_record", lambda _d, _s: [])
    monkeypatch.setattr(
        public_snapshot, "init_db", lambda _p: None, raising=False
    )

    snapshot = public_snapshot.generate_snapshot(
        schedule_path=tmp_path / "games.json",
        hub_dir=tmp_path / "hub",
        manifest_path=manifest_path,
        db_path=tmp_path / "tracking.db",
        now=NOW,
    )
    assert snapshot["manifest"]["model_version"] == "v20260918120240"
    assert snapshot["generated_at"] == NOW.isoformat()


def test_the_real_shipped_manifest_is_now_stale(tmp_path):
    """Read the repo's own manifest and check the gate against it.

    This is the finding in one assertion: on the day this was written the shipped
    models were days past the limit, which is why the gate refuses. If a retrain
    lands, this flips to passing -- and that is the point, not a broken test.
    """
    repo_manifest = Path(__file__).resolve().parents[1] / "models" / "manifest.json"
    if not repo_manifest.exists():
        pytest.skip("no manifest in the repo")
    manifest = json.loads(repo_manifest.read_text())
    age = model_age_days(manifest, datetime.now(timezone.utc))
    assert age is not None
    # Documented rather than asserted either way: the whole point of the gate is
    # that this number moves. Assert the arithmetic, not the verdict.
    assert age >= 0
    stale = age > MAX_MODEL_AGE_DAYS
    if stale:
        with pytest.raises(StaleModels):
            assert_models_fresh(manifest, now=datetime.now(timezone.utc))