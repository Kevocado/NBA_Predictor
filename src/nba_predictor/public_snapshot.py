"""Refuse to publish a snapshot built on stale models (spec section 10, gap G6).

The snapshot carried whatever manifest it was handed. On 2026-10-04 it happily
published a snapshot stamped fresh while serving models trained on 2026-09-18 --
the reader sees "generated today" and reasonably concludes the numbers behind it
are current. They were not, and nothing on the page said so.

The fix is the NFL pattern (`public_snapshot.py` L226-272): compare the model's
`trained_at` against now at publish time and refuse when it is too old.

`MAX_MODEL_AGE_DAYS` is a **named constant because Kevin decides it**. It is not
tuned so a particular model passes; `tests/test_snapshot_staleness.py` pins the
Phase A measurement against it so loosening it fails a test rather than quietly
reversing a conclusion.

A refusal raises rather than warns. The alternative is a snapshot with a
staleness banner on it, which is one more thing a reader has to notice, and the
whole point is that they would not.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from nba_predictor import config
from nba_predictor.services.hub_service import compute_track_record, load_hub_cache
from nba_predictor.services.schedule_repository import load_schedule
from nba_predictor.tracking.store import init_db

#: How old a model may be at publish time. Fourteen days: the daily refresh runs
#: every 24h, so a model older than a fortnight means the refresh is broken --
#: which is exactly what happened for the 16 days between 2026-09-18 and
#: 2026-10-04, and nothing noticed.
MAX_MODEL_AGE_DAYS = 14

#: Clock-skew tolerance for a `trained_at` in the future. A retrain and a
#: publish on machines whose clocks disagree can produce a stamp a few seconds
#: ahead; that is not the same as a stamp days ahead, which is either a clock
#: fault or a manifest written by something that never trained.
MAX_CLOCK_SKEW = timedelta(minutes=5)


class StaleModels(Exception):
    """The snapshot would publish numbers from a model that is too old.

    Carries the age so the log says how far past the line it is, which is the
    difference between "the refresh is broken" and "the threshold is wrong".
    """


def _parse_ts(value) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def model_age_days(manifest: dict, now: datetime | None = None) -> float | None:
    """Days since the manifest's `trained_at`, or None when there is no stamp.

    None is not 0.0: a manifest with no readable `trained_at` carries no evidence
    that it is current, and reporting it as brand new would wave exactly the
    failure this gate exists to catch straight through.
    """
    trained_at = _parse_ts((manifest or {}).get("trained_at"))
    if trained_at is None:
        return None
    now = now or datetime.now(timezone.utc)
    return (now - trained_at).total_seconds() / 86400.0


def assert_models_fresh(
    manifest: dict | None,
    now: datetime | None = None,
    max_age_days: int = MAX_MODEL_AGE_DAYS,
) -> float:
    """Raise `StaleModels` unless the manifest's models are recent enough.

    Returns the age in days so a caller can log it, or None when there is no
    manifest at all.

    **No manifest is not stale.** A repository that has never trained publishes
    `"manifest": null` and the frontend shows a "not trained" state; refusing
    that would break a state the site legitimately has and that misleads no
    reader. What the gate exists for is the opposite case -- a manifest that
    looks current because the snapshot is fresh while the model behind it is
    not. An *undated* manifest is refused, because that one is indistinguishable
    from a fresh one.
    """
    if manifest is None:
        return None

    age = model_age_days(manifest, now)
    if age is not None and age < -MAX_CLOCK_SKEW.total_seconds() / 86400.0:
        # A future stamp yields a NEGATIVE age, which every "is it too old?"
        # comparison passes -- so a manifest dated next month would keep
        # publishing for a month plus the limit. Refuse rather than clamp.
        raise StaleModels(
            f"manifest trained_at={manifest.get('trained_at')} is in the future "
            f"({-age:.1f} days ahead). That is a clock fault or a manifest that "
            "was never really trained on; refusing rather than treating it as "
            "brand new."
        )
    if age is None:
        raise StaleModels(
            f"manifest has no readable trained_at (model_version="
            f"{manifest.get('model_version')!r}); refusing to publish, because a "
            "snapshot with an undated model is indistinguishable from a fresh one"
        )
    if age > max_age_days:
        raise StaleModels(
            f"models are {age:.1f} days old (trained_at="
            f"{manifest.get('trained_at')}), past the {max_age_days}-day limit; "
            "the daily refresh is not running. Refusing to publish a snapshot "
            "whose numbers would look current."
        )
    return age


def generate_snapshot(
    schedule_path: Path,
    hub_dir: Path,
    manifest_path: Path,
    db_path: Path,
    now: datetime | None = None,
    max_age_days: int = MAX_MODEL_AGE_DAYS,
) -> dict:
    """Build the snapshot, refusing to build one from stale models.

    The check is here rather than in `main()` so every caller gets it -- the CLI,
    and anything that imports this function.
    """
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    assert_models_fresh(manifest, now=now, max_age_days=max_age_days)
    schedule = load_schedule(schedule_path)

    return {
        "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
        "schedule": schedule,
        "hub": {
            "teams": load_hub_cache(hub_dir / "teams.json"),
            "players": load_hub_cache(hub_dir / "players.json"),
            "rankings": load_hub_cache(hub_dir / "rankings.json"),
            "standings": load_hub_cache(hub_dir / "standings.json"),
        },
        "track_record": [record.model_dump() for record in compute_track_record(db_path, schedule)],
        "manifest": manifest,
    }

def write_snapshot(snapshot: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))


def main() -> None:
    init_db(config.TRACKING_DB_PATH)
    snapshot = generate_snapshot(
        schedule_path=config.DATA_DIR / "cache" / "schedule" / "games.json",
        hub_dir=config.DATA_DIR / "cache" / "hub",
        manifest_path=config.PROJECT_ROOT / "models" / "manifest.json",
        db_path=config.TRACKING_DB_PATH,
    )
    write_snapshot(snapshot, config.DATA_DIR / "public_snapshot.json")


if __name__ == "__main__":
    main()
