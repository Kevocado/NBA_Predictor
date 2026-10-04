"""The odds refresher: a loop, because nothing else calls it.

Kevin, 2026-10-04: *"call it on a schedule because nba games are spread throughout
the week."*

The thing being scheduled already existed and had never run on its own.
`POST /refresh-odds` is `Depends(require_admin)`, so it is reachable by hand and
by nothing else: there is no scheduler in the app, no cron in the workflows, and
no caller in `main.py`. The deployed container held `SPORTSBOOK_API_KEY` and had
an empty `cache/sportsbook/` — which is what "wired up but never switched on"
looks like from the outside.

## Why a thread and not a cron hitting the endpoint

A GitHub Actions cron would need the admin credential in the workflow, and would
make a metered third-party feed's refresh cadence depend on GitHub's scheduler.
The loop lives with the feature instead, takes its cadence from the feed's own
cache TTL, and costs nothing when the cache is warm.

## Why the cadence is the cache TTL and not something shorter

`data/sportsbook_api.py` caches each response to disk for
`CACHE_TTL_SECONDS` and `_is_fresh` short-circuits the request. So calling this
more often than the TTL spends **no** requests — it re-reads JSON — and calling
it less often serves staler lines than the cache already holds. Tying the tick to
the TTL is therefore the only cadence that is neither free-but-pointless nor
slower than the data underneath it. `NBA` has no bulk odds endpoint (one request
per event), so the per-refresh cost is real and the metered budget matters; see
the module docstring in `data/sportsbook_api.py`.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from nba_predictor.data import sportsbook_api

#: The tick. `sportsbook_api.CACHE_TTL_SECONDS` rather than a number written
#: here: a cadence shorter than the cache spends nothing and achieves nothing,
#: and one longer than it serves staler lines than the cache already holds. Read
#: from the module so a change to the TTL moves this with it.
REFRESH_INTERVAL_SECONDS = sportsbook_api.CACHE_TTL_SECONDS


def _default_interval() -> int:
    return REFRESH_INTERVAL_SECONDS


def start_odds_refresher(
    db_path: Path,
    schedule_path: Path,
    *,
    margin_std: float,
    total_std: float,
    refresh=None,
    load=None,
    clock=time.monotonic,
    sleep=time.sleep,
    should_stop=None,
    interval_seconds: int | None = None,
) -> threading.Thread:
    """Refresh the odds-derived market predictions now, then on the cache TTL.

    A daemon thread started from the app's lifespan, shaped like
    `api.routes.start_mae_warmer`: serving begins whether or not the first
    refresh finishes, and a refresh that hangs cannot hold shutdown. ``refresh``,
    ``load``, ``clock``, ``sleep`` and ``should_stop`` are injected so the loop is
    testable without waiting on a real six-hour tick or spending a request.

    The first pass runs immediately rather than after one interval. A schedule
    that only started refreshing six hours after boot would leave the first day
    of every deploy with no lines at all, which is the state this exists to fix.

    Returns the thread, so a caller can join or inspect it.
    """
    from nba_predictor.services.schedule_repository import load_schedule

    db_path = Path(db_path)
    if load is None:
        load = load_schedule
    if refresh is None:
        from nba_predictor.pipeline.refresh_odds import refresh_market_predictions

        refresh = refresh_market_predictions
    interval = _default_interval() if interval_seconds is None else interval_seconds

    def _run() -> None:
        while should_stop is None or not should_stop():
            try:
                schedule = load(schedule_path)
                refresh(schedule, db_path, margin_std=margin_std, total_std=total_std)
            except Exception:  # noqa: BLE001 - a refresh that fails must not kill the loop
                # Logged and swallowed, like the MAE warmer: a feed that is down
                # or a schedule that cannot be read leaves yesterday's lines in
                # place, which is stale but honest. The alternative — a loop that
                # dies on the first transient error — silently stops refreshing
                # for the life of the container.
                import logging

                logging.getLogger(__name__).exception(
                    "odds refresh failed; keeping the previous lines"
                )
            sleep(interval)

    thread = threading.Thread(target=_run, name="odds-refresher", daemon=True)
    thread.start()
    return thread