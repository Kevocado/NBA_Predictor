"""The player-props aggregate cache on /hub/track-record.

Kevin accepted a few minutes of staleness on this route (NBA_Predictor#48) to
get it under 0.25s warm. The bound is a 300s TTL, the value is warmed at
startup, and the clock is injected so every property below is tested without
sleeping and without touching 400k rows.

Four properties, one per test below:

  * a second call inside the TTL does not touch the database;
  * a call past the TTL recomputes;
  * a recompute that FAILS keeps serving the last good value, and says how old
    it is -- this is the one that differs from `routes._MAE_CACHE`, which drops
    the entry and raises instead;
  * only the aggregate is cached -- the per-pick tables and the counted record
    stay live, because they answer "what did we say, and when".

And the cold-failure case the last of those depends on: with nothing cached
there is no last good value, so the error propagates rather than inventing one.
"""

from pathlib import Path

import pytest

from nba_predictor.api.schemas import TrackRecordOut
from nba_predictor.services import hub_service


class Clock:
    """A clock the test moves by hand. `time.monotonic` is the default in
    production; nothing here waits on real time."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t


def value(total: int = 10) -> TrackRecordOut:
    return TrackRecordOut(
        market="player_props", total_predictions=total, correct_predictions=0, hit_rate=0.0,
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    hub_service.clear_player_props_cache()
    yield
    hub_service.clear_player_props_cache()


@pytest.fixture(autouse=True)
def _restore_ttl():
    original = hub_service.PLAYER_PROPS_CACHE_TTL_SECONDS
    yield
    hub_service.PLAYER_PROPS_CACHE_TTL_SECONDS = original


def test_a_second_call_inside_the_ttl_does_not_touch_the_database():
    """The whole point of the cache: one aggregate per TTL, not one per request."""
    hub_service.PLAYER_PROPS_CACHE_TTL_SECONDS = 300.0
    clock = Clock()
    calls = []

    def compute():
        calls.append(1)
        return value()

    first = hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)
    assert len(calls) == 1

    # 299s later -- still inside the bound.
    clock.advance(299)
    second = hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)

    assert len(calls) == 1, "the database was read again inside the TTL"
    assert second.total_predictions == first.total_predictions


def test_a_call_past_the_ttl_recomputes():
    hub_service.PLAYER_PROPS_CACHE_TTL_SECONDS = 300.0
    clock = Clock()
    calls = []

    def compute():
        calls.append(1)
        return value(total=len(calls) * 100)

    hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)
    assert len(calls) == 1

    # 301s later -- past the bound.
    clock.advance(301)
    refreshed = hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)

    assert len(calls) == 2, "an expired cache was not refreshed"
    assert refreshed.total_predictions == 200, "the refreshed value was not served"


def test_a_failed_recompute_serves_the_last_good_value_and_says_how_old_it_is():
    """The disclosure rule. The failure is reported AND the age travels with the
    value, so the site can print it rather than serving a stale figure silently.
    """
    hub_service.PLAYER_PROPS_CACHE_TTL_SECONDS = 300.0
    clock = Clock()
    good = value(total=42)
    state = {"fail": False}

    def compute():
        if state["fail"]:
            raise RuntimeError("the database went away")
        return good

    hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)
    assert state["fail"] is False

    state["fail"] = True
    clock.advance(600)
    served = hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)

    assert served is not None
    assert served.total_predictions == 42, "the last good value was not kept"
    assert served.served_stale_seconds == pytest.approx(600.0), (
        "a stale value was served without saying how stale"
    )


def test_a_fresh_cache_hit_is_not_called_stale():
    """An ordinary in-TTL hit is FRESH, not stale. Reporting it as stale would
    train a reader to ignore the field, which is how a real staleness notice
    stops being read at all."""
    hub_service.PLAYER_PROPS_CACHE_TTL_SECONDS = 300.0
    clock = Clock()

    def compute():
        return value()

    hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)
    clock.advance(299)
    hit = hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)

    assert hit.served_stale_seconds is None


def test_a_cold_failure_propagates_rather_than_inventing_a_figure():
    """With nothing cached there is no last good value. Returning a zeroed
    TrackRecordOut here would be a number nobody computed."""
    clock = Clock()

    def compute():
        raise RuntimeError("the database went away")

    with pytest.raises(RuntimeError, match="database went away"):
        hub_service.cached_player_props(Path("t.db"), now=clock, compute=compute)


def test_the_cached_value_is_never_mutated_by_a_consumer():
    """`compute_track_record` fills the weekly table onto the row it is given.
    Cached or not, that must not write through to the cached object, or the
    next request would inherit this request's weekly rows."""
    hub_service.PLAYER_PROPS_CACHE_TTL_SECONDS = 300.0
    clock = Clock()

    hub_service.cached_player_props(Path("t.db"), now=clock, compute=lambda: value())

    first = hub_service.cached_player_props(Path("t.db"), now=clock, compute=lambda: value())
    first.total_predictions = 999

    second = hub_service.cached_player_props(Path("t.db"), now=clock, compute=lambda: value())
    assert second.total_predictions == 10, "a consumer's mutation reached the cache"


def test_only_the_aggregate_is_cached_and_the_record_stays_live(tmp_path):
    """Per-pick tables and the counted record are recomputed every request.

    The aggregate answers "how wrong have we been across the season" and is the
    expensive half. The per-pick list and the record answer "what did we say,
    and when" -- questions about picks still being made -- so caching those would
    show a reader a record that stops moving while the page says it is live.
    """
    from nba_predictor.tracking import store

    db_path = tmp_path / "t.db"
    store.init_db(db_path)

    def game(game_id: str) -> dict:
        return {"game_id": game_id, "completed": True, "home_team": "LAL", "away_team": "BOS",
                "home_pts": 110, "away_pts": 100, "game_date": "2026-01-05"}

    def seed(game_id: str) -> None:
        with store.get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO predictions (game_id, created_at, model_version, "
                "home_win_prob, predicted_margin, predicted_total) "
                "VALUES (?,?,?,?,?,?)",
                (game_id, "2026-01-01T00:00:00+00:00", "v1", 0.6, 4.0, 220.0),
            )
            conn.commit()

    hub_service.clear_player_props_cache()
    seed("g1")
    first = {r.market: r for r in hub_service.compute_track_record(db_path, [game("g1")])}

    # A second game is recorded and the schedule grows with it. The counted
    # record answers "what have we called, and how often", so it must move on the
    # next request -- inside any TTL, with or without a warm.
    seed("g2")
    second = {r.market: r for r in hub_service.compute_track_record(db_path, [game("g1"), game("g2")])}

    assert first["game_outcome"].total_predictions == 1
    assert second["game_outcome"].total_predictions == 2, (
        "the counted record was served from the aggregate's cache"
    )
    # And the per-pick disclosure list, which is the same class of question,
    # grows with it.
    assert len(first["game_outcome"].per_pick) == 1
    assert len(second["game_outcome"].per_pick) == 2