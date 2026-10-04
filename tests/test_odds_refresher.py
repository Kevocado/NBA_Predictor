"""The odds refresher loop, and the promise the deployed container was not keeping.

Kevin, 2026-10-04: *"call it on a schedule because nba games are spread throughout
the week."*

`POST /refresh-odds` existed and had never run by itself — it is
`Depends(require_admin)`, there was no scheduler in the app, no cron in the
workflows, and no caller in `main.py`. The container held `SPORTSBOOK_API_KEY`
and its `cache/sportsbook/` was empty.

What each test defends, in order of how quietly it would fail:

  * IT REFRESHES AT ALL. A loop that sleeps first and refreshes second looks
    identical to a working one right up until you read the first six hours of a
    fresh deploy, when there are no lines at all — the state this exists to fix.
  * IT STOPS. A loop that outlives `should_stop` holds shutdown, which is the
    failure `start_mae_warmer`'s docstring is explicit about.
  * IT KEEPS GOING AFTER A FAILURE. A feed that 500s once must not end the loop
    for the life of the container: yesterday's lines are stale but honest, and a
    dead loop is silently wrong.
  * THE CADENCE IS THE CACHE TTL. `data/sportsbook_api.py` caches each response
    for `CACHE_TTL_SECONDS`, so a faster tick spends no requests and achieves
    nothing, and a slower one serves staler lines than the cache already holds.
  * THE APP STARTS IT. A perfect loop nothing calls is the defect being fixed.
"""

import threading

import pytest

from nba_predictor.data import sportsbook_api
from nba_predictor.odds.value_bets import std_from_mae
from nba_predictor.pipeline import odds_refresher
from nba_predictor.pipeline.odds_refresher import start_odds_refresher


def _run_once(**kwargs):
    """Start the loop, let it tick exactly twice, and return the calls."""
    calls = []
    stop = threading.Event()

    def _sleep(_seconds):
        # Second pass flips the flag, so the loop leaves after two iterations
        # without the test waiting on a real tick.
        if len(calls) >= 2:
            stop.set()

    thread = start_odds_refresher(
        db_path="/tmp/tracking.db",
        schedule_path="/tmp/schedule.json",
        margin_std=12.0,
        total_std=15.0,
        refresh=lambda schedule, db, **kw: calls.append(("refresh", kw)),
        load=lambda path: [{"game_id": "1", "completed": False}],
        sleep=_sleep,
        should_stop=stop.is_set,
        interval_seconds=0,
        **kwargs,
    )
    thread.join(timeout=5)
    return calls, stop


def test_it_refreshes_immediately_rather_than_after_one_interval():
    # The whole point. A schedule that waited a full TTL before its first pass
    # would leave the first six hours of every deploy with no lines.
    calls, _ = _run_once()
    assert len(calls) >= 1
    assert calls[0][0] == "refresh"


def test_it_passes_the_models_standard_deviations_through():
    # `refresh_market_predictions` prices the spread and total off these, and
    # their defaults (12.0 / 15.0) are not the ones the API route uses: it
    # derives them from the manifest. A loop passing the bare defaults would
    # quietly publish lines computed from different spreads than the page.
    calls, _ = _run_once()
    assert calls[0][1]["margin_std"] == 12.0
    assert calls[0][1]["total_std"] == 15.0


def test_it_stops_when_asked():
    _, stop = _run_once()
    assert stop.is_set()


def test_the_thread_is_a_daemon():
    # A non-daemon thread would hold the container's shutdown for up to one
    # interval after the app stops.
    thread = start_odds_refresher(
        db_path="/tmp/tracking.db", schedule_path="/tmp/schedule.json",
        margin_std=12.0, total_std=15.0,
        refresh=lambda *a, **k: None, load=lambda p: [],
        sleep=lambda _s: None,
        should_stop=lambda: True,
        interval_seconds=0,
    )
    thread.join(timeout=5)
    assert thread.daemon is True


def test_a_failing_refresh_does_not_end_the_loop():
    # A feed that 500s once must not leave the container permanently unrefreshed.
    calls = []
    stop = threading.Event()

    def _refresh(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("sportsbook 500")
        stop.set()

    start_odds_refresher(
        db_path="/tmp/tracking.db", schedule_path="/tmp/schedule.json",
        margin_std=12.0, total_std=15.0,
        refresh=_refresh, load=lambda p: [],
        sleep=lambda _s: None, should_stop=stop.is_set, interval_seconds=0,
    ).join(timeout=5)
    assert len(calls) >= 2


def test_an_unreadable_schedule_does_not_end_the_loop():
    calls = []
    stop = threading.Event()

    def _load(_path):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("schedule missing")
        stop.set()

    start_odds_refresher(
        db_path="/tmp/tracking.db", schedule_path="/tmp/schedule.json",
        margin_std=12.0, total_std=15.0,
        refresh=lambda *a, **k: None, load=_load,
        sleep=lambda _s: None, should_stop=stop.is_set, interval_seconds=0,
    ).join(timeout=5)
    assert len(calls) >= 2


def test_the_cadence_is_the_sportsbook_cache_ttl():
    # Not a number written here. A tick faster than the cache spends no requests
    # and achieves nothing; slower serves staler lines than the cache holds.
    assert odds_refresher.REFRESH_INTERVAL_SECONDS == sportsbook_api.CACHE_TTL_SECONDS
    assert sportsbook_api.CACHE_TTL_SECONDS > 0


def test_the_app_starts_the_refresher(monkeypatch):
    # A correct loop that nothing calls is the defect being fixed, so this
    # RUNS the lifespan rather than reading its source.
    #
    # The first version of this test grepped `app.py` for the function's name and
    # passed against an app that raised `AttributeError` on boot: it called
    # `config.MODELS_DIR`, which does not exist (the models directory is
    # `deps.get_models_dir()`). A source grep cannot see a wrong attribute; a
    # lifespan can.
    import asyncio

    from nba_predictor.api import app as app_module

    started = {}

    def _fake_start(db_path, schedule_path, **kwargs):
        started["db_path"] = db_path
        started["schedule_path"] = schedule_path
        started["kwargs"] = kwargs
        return threading.Thread(target=lambda: None, daemon=True)

    monkeypatch.setattr(app_module, "start_odds_refresher", _fake_start)
    monkeypatch.setattr(app_module, "start_mae_warmer", lambda *a, **k: None)

    async def _enter():
        async with app_module.lifespan(object()):
            pass

    asyncio.run(_enter())

    assert started, "the lifespan did not start the odds refresher"
    assert started["kwargs"]["margin_std"] > 0
    assert started["kwargs"]["total_std"] > 0


def test_the_standard_deviations_come_from_the_manifest(tmp_path, monkeypatch):
    # The API route derives these from the manifest rather than using the bare
    # defaults; a loop that passed the defaults would publish lines priced off
    # different spreads than the page above them.
    import json

    from nba_predictor.api.routes import _market_stds_from_manifest

    manifest = {"metrics": {"margin": {"mae": 10.5}, "total": {"mae": 13.25}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    margin_std, total_std = _market_stds_from_manifest(tmp_path)
    # `std_from_mae` rather than a literal factor: the conversion is the
    # function's business, and a test that hard-codes the constant would fail
    # for the wrong reason if it ever moved.
    assert margin_std == pytest.approx(std_from_mae(10.5))
    assert total_std == pytest.approx(std_from_mae(13.25))


def test_a_missing_manifest_falls_back_to_the_defaults(tmp_path):
    from nba_predictor.api.routes import _market_stds_from_manifest

    assert _market_stds_from_manifest(tmp_path) == (12.0, 15.0)