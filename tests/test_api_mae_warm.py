"""The MAE cache is warmed off the request path, and never served past its age.

/games/{game_id}/players prints an in-sample MAE beside every projection, and
that MAE is a property of the tracking database, not of the request. Computing
it costs about a second of bulk reads against the 18 MB tracking.db and the
1,760-game schedule cache, and it used to be charged to whichever visitor
arrived first after the database changed.

What is pinned here, because each is a way this can quietly go wrong:

  * warming happens at startup, in a thread, so no visitor pays it;
  * warming cannot take the API down, and cannot block serving;
  * an entry has an explicit age bound, and past it the value is RECOMPUTED,
    never served -- the injury-report lesson, on this cache;
  * a database change re-warms, reusing the (mtime, size) state the cache
    already keyed on, so no new notion of "the DB changed" is invented;
  * a failed warm logs loudly and leaves no stale entry behind.

Every clock here is injected. No test sleeps.
"""

import json
import threading
import time

import pytest

from nba_predictor.api import routes


class FakeClock:
    """A monotonic clock a test moves by hand."""

    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def _clean_cache():
    routes._MAE_CACHE.clear()
    yield
    routes._MAE_CACHE.clear()


def _db(tmp_path, content=b"x"):
    db = tmp_path / "tracking.db"
    db.write_bytes(content)
    return db


def _schedule_file(tmp_path):
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps([{"game_id": "g1"}]))
    return schedule_path


def _counting(monkeypatch, result=None, raises=None):
    """resolved_player_props stand-in that records every call."""
    calls = []

    def _resolved(db_path, schedule):
        calls.append(1)
        if raises is not None:
            raise raises
        return result if result is not None else [
            # `made_before_tip` is carried on every row the real
            # `resolved_player_props` returns, so a stand-in that omits it is
            # not a stand-in -- routes._mae_record reads it to split the
            # headline from the pre-tip figure.
            {"stat": "points", "predicted_value": 20.0, "actual_value": 24.0,
             "made_before_tip": True},
        ]

    monkeypatch.setattr(routes, "resolved_player_props", _resolved)
    return calls


# ---------------------------------------------------------------- warming


def test_warm_mae_cache_populates_the_cache_so_no_request_has_to(tmp_path, monkeypatch):
    db = _db(tmp_path)
    calls = _counting(monkeypatch)
    schedule = [{"game_id": "g1"}]

    assert routes.warm_mae_cache(db, schedule) is True
    assert len(calls) == 1

    # The next three requests are served from the warm entry.
    for _ in range(3):
        routes._mae_by_stat(db, schedule)
    assert len(calls) == 1, "a warmed MAE was recomputed on the request path"


def test_a_failed_warm_reports_failure_and_leaves_no_stale_entry(tmp_path, monkeypatch, caplog):
    db = _db(tmp_path)
    _counting(monkeypatch, raises=RuntimeError("tracking.db is locked"))

    with caplog.at_level("ERROR"):
        assert routes.warm_mae_cache(db, [{"game_id": "g1"}]) is False

    assert routes._MAE_CACHE == {}, "a failed warm cached something anyway"
    assert any("MAE" in r.message for r in caplog.records), "a failed warm failed silently"


def test_a_failed_warm_does_not_take_the_api_down(tmp_path, monkeypatch):
    """The endpoint still answers after a failed warm: the request path
    recomputes for itself, and a failure there is one request's error, not an
    outage."""
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import create_app
    from nba_predictor.api import deps
    from nba_predictor.tracking import store

    db = tmp_path / "tracking.db"
    store.init_db(db)
    schedule_path = tmp_path / "games.json"
    schedule_path.write_text(json.dumps([
        {"game_id": "g1", "game_date": "2026-11-01", "home_team": "BOS", "away_team": "MIA"}
    ]))
    schedule = [{"game_id": "g1", "game_date": "2026-11-01"}]

    def _locked(db_path, sched):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(routes, "resolved_player_props", _locked)
    assert routes.warm_mae_cache(db, schedule) is False
    monkeypatch.setattr(routes, "resolved_player_props", lambda *a: [])

    app = create_app()
    app.dependency_overrides[deps.get_db_path] = lambda: db
    app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
    app.dependency_overrides[deps.get_injury_report] = lambda: []

    try:
        client = TestClient(app)
        assert client.get("/games/g1/players").status_code == 200
        assert client.get("/health").status_code == 200
    finally:
        app.dependency_overrides.clear()


# ------------------------------------------------------------- the age bound


def test_an_entry_within_the_age_bound_is_served(tmp_path, monkeypatch):
    db = _db(tmp_path)
    calls = _counting(monkeypatch)
    clock = FakeClock()
    schedule = [{"game_id": "g1"}]

    routes._mae_by_stat(db, schedule, now=clock)
    clock.advance(routes.MAE_CACHE_TTL_SECONDS - 1)
    routes._mae_by_stat(db, schedule, now=clock)

    assert len(calls) == 1


def test_an_entry_past_the_age_bound_is_not_served(tmp_path, monkeypatch):
    """The injury-report bug in this repo's own history: a cache with no age
    check froze one snapshot forever. Past the bound the value is recomputed,
    and the NEW answer is what a request sees."""
    db = _db(tmp_path)
    seen = []

    def _resolved(db_path, schedule):
        seen.append(1)
        return [{"stat": "points", "predicted_value": 20.0, "actual_value": 20.0 + len(seen),
                 "made_before_tip": True}]

    monkeypatch.setattr(routes, "resolved_player_props", _resolved)
    clock = FakeClock()
    schedule = [{"game_id": "g1"}]

    first = routes._mae_by_stat(db, schedule, now=clock)
    clock.advance(routes.MAE_CACHE_TTL_SECONDS + 1)
    second = routes._mae_by_stat(db, schedule, now=clock)

    assert len(seen) == 2, "the stale entry was served past its age bound"
    assert first["points"] == pytest.approx(1.0)
    assert second["points"] == pytest.approx(2.0)


def test_an_expired_entry_is_dropped_even_when_the_rewarm_fails(tmp_path, monkeypatch):
    """Failing loudly beats serving stale: if the entry is past its bound and
    the recompute fails, the request gets the failure, not the expired value."""
    db = _db(tmp_path)
    monkeypatch.setattr(routes, "resolved_player_props", lambda *a: [
        {"stat": "points", "predicted_value": 20.0, "actual_value": 21.0,
         "made_before_tip": True}
    ])
    clock = FakeClock()
    schedule = [{"game_id": "g1"}]
    routes._mae_by_stat(db, schedule, now=clock)

    def _boom(db_path, schedule_):
        raise RuntimeError("tracking.db is locked")

    monkeypatch.setattr(routes, "resolved_player_props", _boom)
    clock.advance(routes.MAE_CACHE_TTL_SECONDS + 1)

    with pytest.raises(RuntimeError):
        routes._mae_by_stat(db, schedule, now=clock)
    assert routes._MAE_CACHE == {}, "an entry that could not be refreshed stayed cached"


def test_the_age_bound_is_bounded_and_configurable():
    # Fifteen minutes, the same bound as the injury report: long against the
    # write cadence that invalidates this cache anyway, short enough that a
    # silently-missed database change cannot be served forever.
    assert routes.MAE_CACHE_TTL_SECONDS == 900
    assert 0 < routes.MAE_CACHE_TTL_SECONDS <= 3600


# --------------------------------------------------- the database-change hook


def test_the_warmer_warms_once_and_leaves_an_unchanged_database_alone(tmp_path, monkeypatch):
    """The existing invalidation signal is (mtime_ns, size) of the tracking db --
    routes._db_state, the key this cache already keys on. The warmer watches it,
    so an idle database costs one warm at startup and nothing after."""
    db = _db(tmp_path)
    calls = _counting(monkeypatch)
    schedule_path = _schedule_file(tmp_path)
    clock = FakeClock()

    thread = routes.start_mae_warmer(
        db,
        schedule_path,
        clock=clock,
        sleep=lambda s: clock.advance(routes.MAE_WARM_POLL_SECONDS),
        should_stop=lambda: clock.now >= 1000 + 5 * routes.MAE_WARM_POLL_SECONDS,
    )
    thread.join(timeout=10)

    assert not thread.is_alive(), "the warmer loop did not terminate"
    assert len(calls) == 1, f"an unchanged database was re-warmed {len(calls) - 1} extra times"


def test_a_changed_database_is_re_warmed_by_the_poll_loop(tmp_path, monkeypatch):
    db = _db(tmp_path)
    calls = _counting(monkeypatch)
    schedule_path = _schedule_file(tmp_path)
    clock = FakeClock()
    changed = {"done": False}

    def _sleep(seconds):
        clock.advance(routes.MAE_WARM_POLL_SECONDS)
        if not changed["done"]:
            # A game resolves while the warmer is asleep.
            db.write_bytes(b"changed contents")
            changed["done"] = True

    thread = routes.start_mae_warmer(
        db, schedule_path, clock=clock, sleep=_sleep, should_stop=lambda: len(calls) >= 2
    )
    thread.join(timeout=10)

    assert not thread.is_alive(), "the warmer loop did not terminate"
    assert len(calls) == 2, "a changed tracking database did not trigger a re-warm"


def test_the_database_write_path_rewarms_immediately(tmp_path, monkeypatch):
    """run_ingest is what writes predictions and outcomes, so it is the moment
    the MAE changes. It re-warms on the spot rather than waiting for the poll."""
    calls = _counting(monkeypatch)
    db = _db(tmp_path)

    routes.note_database_changed(db, [{"game_id": "g1"}])

    assert len(calls) == 1, "a database write did not trigger a re-warm"


def test_a_write_whose_rewarm_fails_leaves_the_pre_write_mae_in_place(tmp_path, monkeypatch):
    """It must not: the old number would then read as current."""
    db = _db(tmp_path)
    monkeypatch.setattr(routes, "resolved_player_props", lambda *a: [
        {"stat": "points", "predicted_value": 20.0, "actual_value": 21.0,
         "made_before_tip": True}
    ])
    schedule = [{"game_id": "g1"}]
    routes.warm_mae_cache(db, schedule)
    assert routes._MAE_CACHE

    db.write_bytes(b"changed")
    monkeypatch.setattr(
        routes,
        "resolved_player_props",
        lambda *a: (_ for _ in ()).throw(RuntimeError("tracking.db is locked")),
    )
    assert routes.warm_mae_cache(db, schedule) is False
    assert routes._MAE_CACHE == {}


def test_a_request_arriving_during_the_warm_does_not_start_a_second_compute(tmp_path, monkeypatch):
    """The startup warm and the first visitor used to read the same rows at the
    same time and serialise on the tracking store's lock, which measured slower
    than either alone. The request waits for the warm and reuses its result."""
    db = _db(tmp_path)
    schedule = [{"game_id": "g1"}]
    calls = _counting(monkeypatch)
    inside = threading.Event()
    release = threading.Event()

    def _slow(db_path, sched):
        calls.append(1)
        inside.set()
        release.wait(timeout=10)
        return [{"stat": "points", "predicted_value": 20.0, "actual_value": 24.0,
                 "made_before_tip": True}]

    monkeypatch.setattr(routes, "resolved_player_props", _slow)

    warm = threading.Thread(target=routes.warm_mae_cache, args=(db, schedule), daemon=True)
    warm.start()
    assert inside.wait(timeout=10), "the warm never started computing"

    request = threading.Thread(target=routes._mae_by_stat, args=(db, schedule), daemon=True)
    request.start()
    # The request is now waiting on the warm. Give it long enough that it would
    # have finished a duplicate read if it were doing one.
    request.join(timeout=1)
    assert request.is_alive(), "the request duplicated the in-flight compute instead of waiting"

    release.set()
    request.join(timeout=10)
    warm.join(timeout=10)

    assert len(calls) == 1, f"the same rows were read {len(calls)} times for one cold state"


def test_a_request_is_not_blocked_forever_by_a_warm_that_hangs(tmp_path, monkeypatch):
    """Bounded wait: past it the request computes for itself rather than waiting
    on a warm that will never finish. Serving never depends on the cache."""
    db = _db(tmp_path)
    schedule = [{"game_id": "g1"}]
    hung = threading.Event()
    release = threading.Event()

    calls = []

    def _hang(db_path, sched):
        calls.append(1)
        if len(calls) == 1:  # only the warm hangs; the request's own read answers
            hung.set()
            release.wait(timeout=30)
        return []

    monkeypatch.setattr(routes, "resolved_player_props", _hang)
    monkeypatch.setattr(routes, "MAE_COMPUTE_WAIT_SECONDS", 0.2)

    warm = threading.Thread(target=routes.warm_mae_cache, args=(db, schedule), daemon=True)
    warm.start()
    assert hung.wait(timeout=10), "the warm never started computing"

    started = time.perf_counter()
    routes._mae_by_stat(db, schedule)  # must return on its own compute
    elapsed = time.perf_counter() - started
    release.set()
    warm.join(timeout=10)

    assert len(calls) == 2, "the request did not fall back to computing for itself"

    assert elapsed < 5, f"the request waited {elapsed:.1f}s on a hung warm"


# ------------------------------------------------------- not blocking serving


def test_the_warmer_runs_in_a_daemon_thread_that_serves_during_it(tmp_path, monkeypatch):
    """Serving must not wait on the warm. A compute that blocks still leaves
    /health answering, and the app comes up."""
    from fastapi.testclient import TestClient

    from nba_predictor.api.app import create_app
    from nba_predictor.api import deps

    started = threading.Event()
    release = threading.Event()

    def _blocked(db_path, schedule):
        started.set()
        release.wait(timeout=30)
        return []

    monkeypatch.setattr(routes, "resolved_player_props", _blocked)
    db = _db(tmp_path)
    schedule_path = _schedule_file(tmp_path)

    thread = routes.start_mae_warmer(db, schedule_path, sleep=lambda s: release.wait(timeout=30))
    try:
        assert started.wait(timeout=10), "the warmer never ran"
        assert thread.daemon, "the warmer thread is not a daemon; it could hold shutdown"

        app = create_app()
        app.dependency_overrides[deps.get_db_path] = lambda: db
        app.dependency_overrides[deps.get_schedule_path] = lambda: schedule_path
        app.dependency_overrides[deps.get_injury_report] = lambda: []
        try:
            client = TestClient(app)
            assert client.get("/health").status_code == 200
        finally:
            app.dependency_overrides.clear()
    finally:
        release.set()
        thread.join(timeout=10)


def test_the_app_warms_on_startup_and_a_failed_warm_still_serves(tmp_path, monkeypatch):
    """The startup hook is wired into the app, and it is wired to a failure mode
    that does not stop the app from starting."""
    from fastapi.testclient import TestClient

    from nba_predictor.api import app as app_module
    from nba_predictor.api.app import create_app
    from nba_predictor.api import deps

    seen = []

    def _spy(db_path, schedule_path):
        seen.append((db_path, schedule_path))
        raise RuntimeError("tracking.db is locked")

    monkeypatch.setattr(app_module, "start_mae_warmer", _spy)
    app = create_app()
    app.dependency_overrides[deps.get_injury_report] = lambda: []

    try:
        with TestClient(app) as client:  # entering the context runs the lifespan
            assert len(seen) == 1, "the app did not warm the MAE cache at startup"
            assert client.get("/health").status_code == 200
    finally:
        app.dependency_overrides.clear()
