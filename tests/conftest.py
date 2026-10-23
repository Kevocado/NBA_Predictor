"""Test-suite-wide fixtures.

## Why the box-history warmer is disabled here

`start_box_history_warmer` does real work: it reads the schedule cache and then
the box scores behind it, fetching any that ESPN has not cached yet. That is
right in production -- the ingest pipeline is what populates the cache, and a
daemon thread assembling from it keeps the work off the request path.

It is wrong under pytest. `TestClient(create_app())` runs the app's lifespan, so
the warmer would start from a thread no test controls, mid-suite, and do live
network I/O. That showed up as intermittent failures in
`tests/test_explain_proxy.py` (which builds a temp `frontend/dist` and asserts on
routing) whenever the thread happened to be mid-request.

The other warmers do not need this: `start_mae_warmer` and the odds refresher
read local files only. This one is the anomaly, because the schedule repository
keeps scores and not box scores.

So it is stubbed here, and `tests/test_box_score_history.py` exercises the real
thing directly with its clock, sleep and stop flag injected.
"""
from __future__ import annotations

import threading

import pytest


@pytest.fixture(autouse=True)
def _no_warmers(monkeypatch):
    """The app's lifespan must not start any real warmers under pytest.

    All warmers start daemon threads that do I/O and leave global state.
    They are exercised in their own test files with injected clock/sleep/stop.
    """
    from nba_predictor.api import app as app_module

    def _noop(*args, **kwargs):
        return threading.Thread(target=lambda: None)

    monkeypatch.setattr(app_module, "start_box_history_warmer", _noop)
    monkeypatch.setattr(app_module, "start_mae_warmer", _noop)
    monkeypatch.setattr(app_module, "start_odds_refresher", _noop)
    monkeypatch.setattr(app_module, "start_player_props_refresher", _noop)
    # warm_player_props_cache is called directly in lifespan, not a thread starter.
    # It reads local DB but may create the tracking.db file; stub it to avoid
    # polluting the shipped tracking database during tests.
    monkeypatch.setattr(app_module, "warm_player_props_cache", lambda *a, **k: True)
    return _noop


@pytest.fixture(autouse=True)
def _clean_box_history_memory(monkeypatch):
    """Ensure box-history in-memory cache is empty for each test."""
    from nba_predictor.api import facts as facts_mod
    monkeypatch.setattr(facts_mod, "_BOX_HISTORY", {})
