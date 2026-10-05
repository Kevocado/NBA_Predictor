"""`GET /api/signals/{game_id}` for NBA.

The endpoint's job is four things and no more: find the game, ask each adapter,
keep the strongest few, and never let a signal's absence break the game page.

The one that is NBA-specific is the FIRST input. `/games/{game_id}/players`
withholds an out player's rows, and `OutPlayerOut` carries no projection, no stat
and no rank — so this endpoint cannot be built from that response without inventing
the figure. It reads `picks_by_player_stat` (the full pool, out players included)
and `resolve_out_players`, which is what `get_game_players` reads too, and stops
before the removal.

`get_game_players` is a live FastAPI route with `Depends` on the store, the
schedule and the injury report, so this file calls `signals_for_game` and stubs its
four arguments. The route wrapper is exercised by the app-import test and by CI.

Run: python -m pytest tests/test_signals_endpoint.py -q
"""
from __future__ import annotations

import pytest
from fastapi import HTTPException

from nba_predictor.api import signals as signals_module
from nba_predictor.api.signals import MAX_SIGNALS, get_signals, signals_for_game


def pick(value: float) -> dict:
    return {"pick": {"predicted_value": value}, "rebuilt": False}


OUT = {
    "player_id": "p2", "player_name": "Big", "team": "LAL",
    "status": "out", "source": "ESPN injury report", "dated": "2026-10-04",
}

GAME = {"game_id": "401", "home_team": "LAL", "away_team": "BOS", "tipoff": "2099-01-01T00:00:00Z"}


@pytest.fixture
def wired(monkeypatch):
    """The two store reads, plus the name map and the started flag. Everything
    else here is the endpoint's own logic under test."""
    state = {"picks": {}, "injuries": [], "names": {"p2": "Big"}}

    monkeypatch.setattr(
        signals_module, "get_game",
        lambda schedule, game_id: GAME if game_id == "401" else None,
    )
    monkeypatch.setattr(
        signals_module, "picks_by_player_stat",
        lambda db, game_id, game: state["picks"],
    )
    # `resolve_out_players` is deliberately NOT stubbed. It is a pure function over
    # (injuries, ids) that filters on `availability.is_out`, and stubbing it with a
    # status-blind fake is what made the doubtful-player test below pass for the
    # wrong reason on its first run -- the fake returned an OUT entry for a
    # day-to-day one and the real filter never got a say.
    monkeypatch.setattr(signals_module, "load_player_name_map", lambda path: state["names"])
    # A future tip-off, so the started rule is not what is under test unless a test
    # says so.
    monkeypatch.setattr(signals_module, "_status", lambda game, now: "scheduled")
    monkeypatch.setattr(signals_module, "_now", lambda: None)
    return state


def test_returns_an_absence_for_a_game_with_someone_out(wired):
    wired["picks"] = {("p2", "points"): pick(31.5), ("p3", "points"): pick(24.0)}
    wired["injuries"] = [dict(OUT)]
    payload = get_signals("401", db_path=None, schedule=[GAME], injuries=wired["injuries"])
    assert payload["sport"] == "nba"
    assert payload["id"] == "401"
    assert len(payload["signals"]) == 1
    assert payload["signals"][0]["headline"]["text"] == (
        "Out: Big, our #1 Points projection (31.5 pts)"
    )


def test_an_empty_list_is_a_complete_answer_not_an_error(wired):
    """Spec §2: no data, no signal. Most games have nobody out, so this is the
    common case and it must not look like a failure."""
    wired["picks"] = {("p3", "points"): pick(24.0)}
    wired["injuries"] = []
    assert get_signals("401", db_path=None, schedule=[GAME], injuries=[]) == {
        "sport": "nba", "id": "401", "signals": []
    }


def test_a_started_game_carries_no_signal(wired, monkeypatch):
    """An absence is only information before tip-off. After it, the injury report
    describes a lineup that was already known."""
    wired["picks"] = {("p2", "points"): pick(31.5)}
    wired["injuries"] = [dict(OUT)]
    monkeypatch.setattr(signals_module, "_status", lambda game, now: "final")
    assert get_signals("401", db_path=None, schedule=[GAME], injuries=wired["injuries"])["signals"] == []


def test_an_unknown_game_is_a_404(wired):
    """NOT swallowed into an empty list: an id the client got wrong should say so.
    The id grammar belongs to the schedule, and this router does not own one."""
    with pytest.raises(HTTPException) as exc:
        get_signals("999", db_path=None, schedule=[GAME], injuries=[])
    assert exc.value.status_code == 404


def test_a_404_is_not_absorbed_into_the_success_shape(wired):
    with pytest.raises(HTTPException):
        get_signals("999", db_path=None, schedule=[GAME], injuries=[])


def test_an_adapter_that_raises_does_not_take_down_the_page(wired, monkeypatch):
    """A signal is an enhancement on a game page. The page has to survive it."""
    wired["picks"] = {("p2", "points"): pick(31.5)}
    wired["injuries"] = [dict(OUT)]
    monkeypatch.setattr(
        signals_module.absence, "absence_signal",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("store down")),
    )
    assert get_signals("401", db_path=None, schedule=[GAME], injuries=wired["injuries"])["signals"] == []


def test_a_store_that_raises_still_returns_the_success_shape(wired, monkeypatch):
    """A client reading `sport` or `id` must not get a different object only when
    the backend is failing — the one moment it is least able to cope."""
    def boom(db, game_id, game):
        raise RuntimeError("tracking.db unreadable")

    monkeypatch.setattr(signals_module, "picks_by_player_stat", boom)
    assert get_signals("401", db_path=None, schedule=[GAME], injuries=[]) == {
        "sport": "nba", "id": "401", "signals": []
    }


def test_an_out_player_absent_from_the_pool_yields_no_signal(wired):
    """Injured and absent from the projections: no figure to draw and no rank to
    state, so no row. An empty-state card is the alternative and it says nothing."""
    wired["picks"] = {("p3", "points"): pick(24.0)}
    wired["injuries"] = [{"player_id": "zz", "status": "out"}]
    assert get_signals("401", db_path=None, schedule=[GAME], injuries=wired["injuries"])["signals"] == []


def test_a_doubtful_player_alone_yields_no_signal(wired):
    """`day-to-day` stays a flag and those players STAY in the ranking, with a note
    beside them. Routing one through the out feed would make the shipped frontend
    DELETE somebody, which is why `resolve_doubtful_players` is never merged into it
    on the wire.

    Exercised against the REAL `resolve_out_players`, so this is a test of the
    status filter rather than of a stub that agreed with it.
    """
    wired["picks"] = {("p2", "points"): pick(31.5)}
    wired["injuries"] = [{
        "player_id": "p2", "player_name": "Big", "team": "LAL",
        "status": "day-to-day", "source": "ESPN injury report", "dated": "2026-10-04",
    }]
    assert get_signals("401", db_path=None, schedule=[GAME], injuries=wired["injuries"])["signals"] == []


def test_max_signals_is_the_specs_two_or_three():
    """Spec §2's rule lives at the endpoint. One adapter returns at most one, so
    nothing is truncated — but the cap is the rule, so it is asserted."""
    assert MAX_SIGNALS in (2, 3)


def test_signals_come_back_strongest_first(wired):
    wired["picks"] = {("p2", "points"): pick(31.5), ("p3", "points"): pick(24.0)}
    wired["injuries"] = [dict(OUT)]
    signals = signals_for_game("401", None, [GAME], wired["injuries"])
    assert [s["strength"] for s in signals] == sorted(
        (s["strength"] for s in signals), reverse=True
    )


def test_the_route_is_mounted_on_the_app():
    """A router that exists and is never mounted is a 404 at runtime, and every
    test above calls the function directly.

    Read off `openapi()["paths"]`, not `app.routes`: this app wraps its includes in
    a `_IncludedRouter`, so a route mounted correctly does not appear as an
    `APIRoute` at the top level -- and a test asserting on `app.routes` would fail
    on a correctly mounted router, which is the worse failure.
    """
    from nba_predictor.api import app as app_module

    assert "/api/signals/{game_id}" in app_module.app.openapi()["paths"], (
        "/api/signals/{game_id} is not mounted; the endpoint would 404 in production"
    )
