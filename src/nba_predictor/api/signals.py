"""signals.py — `GET /api/signals/{game_id}`, this game's signal payloads.

Spec `2026-10-01-fixture-signals-design.md` §3 (the contract) and §2: *"no data,
no signal"* — a game with nothing to say returns `{"signals": []}`, which the client
renders as no rows at all, not as an empty state or a filler card.

This is NBA's first signal. The visual lives in the shared component
(`predictor-ui`'s `SignalRows`), which refused `absence_strip` outright until the
hub PR that built it; the payload here is deliberately independent of that, because
an adapter that could not be written and tested before its renderer existed would
mean the two halves could not be checked separately.

## Why this endpoint reads the store and not the API

`/games/{game_id}/players` **withholds** an out player's rows, and
`OutPlayerOut` carries no projection, no stat and no rank. So there is nothing on
the wire to quote — a signal built from that response would have to invent the
figure, which is the one thing this repo must not do.

So this reads the same two sources `get_game_players` reads, in the same order, and
stops before the removal:

    picks_by_player_stat(db, game_id, game)   # the FULL pool, out players included
    resolve_out_players(injuries, pick ids)  # the out feed

Both are ordinary functions on modules this app already imports. The alternative —
adding the projection to `OutPlayerOut` — would put a number on a type whose whole
purpose is to be a removal instruction the shipped frontend applies to every row it
is handed, which is the failure `resolve_doubtful_players`'s docstring warns about.

`MAX_SIGNALS` is the spec's own rule — "a fixed rule (not the model) ranks them by
`strength` and keeps the top 2-3" — so it lives here, at the endpoint, and not in a
component. One adapter returns at most one signal, so nothing is truncated today.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from . import deps
from .availability import resolve_out_players
from .facts import _now, _status
from ..config import DATA_DIR
from ..services.hub_service import load_player_name_map
from ..services.schedule_repository import get_game
from ..signals import absence
from ..tracking.player_props import picks_by_player_stat

router = APIRouter(prefix="/api")
logger = logging.getLogger(__name__)

#: Spec §2. One adapter, so this never truncates anything today.
MAX_SIGNALS = 3


def signals_for_game(
    game_id: str,
    db_path,
    schedule: list[dict],
    injuries: list[dict],
) -> list[dict]:
    """Every signal this game can honestly carry, strongest first.

    Each adapter is asked independently and may return nothing. An adapter that
    raises is treated as "no signal" rather than being allowed to take down the
    endpoint: a signal is an enhancement on a game page, and the page has to
    survive its absence.

    A STARTED game carries none, and that is the rule rather than an
    optimisation — see `facts._status`, which decides it for the facts bundle, and
    `get_game_players`, which is already gated the same way upstream. An absence is
    only information before tip-off; quoting one afterwards is hindsight wearing a
    prediction's clothes, and `pre_kickoff_only` is the same claim made about a
    payload.
    """
    game = get_game(schedule, game_id)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Unknown game: {game_id}")

    if _status(game, _now()) in ("live", "final"):
        return []

    picks = picks_by_player_stat(db_path, game_id, game)
    try:
        signal = absence.absence_signal(
            game_id,
            picks,
            resolve_out_players(injuries, {pid for pid, _ in picks}),
            load_player_name_map(DATA_DIR / "cache" / "hub" / "players.json"),
        )
    except Exception:
        logger.exception("absence signal unavailable for %s", game_id)
        signal = None

    found = [signal] if signal is not None else []
    return sorted(found, key=lambda s: s["strength"], reverse=True)[:MAX_SIGNALS]


@router.get("/signals/{game_id}")
def get_signals(
    game_id: str,
    schedule: list[dict] = Depends(deps.get_schedule),
    db_path=Depends(deps.get_db_path),
    injuries: list[dict] = Depends(deps.get_injury_report),
) -> dict:
    """The signals for one game. `{"signals": []}` is a valid, complete answer.

    An unknown game is a 404, matching `/games/{game_id}`: the id grammar belongs
    to the schedule and this router does not own a second one.
    """
    try:
        signals = signals_for_game(game_id, db_path, schedule, injuries)
    except HTTPException:
        raise
    except Exception:
        # The game itself is unreadable, or the store is down.
        logger.exception("signals unavailable for %s", game_id)
        # The SAME shape as the success path, so a client reading `sport` or `id`
        # does not get a different object only when the backend is failing — the
        # one moment a client is least able to cope with a special case.
        return {"sport": "nba", "id": game_id, "signals": []}

    return {"sport": "nba", "id": game_id, "signals": signals}
