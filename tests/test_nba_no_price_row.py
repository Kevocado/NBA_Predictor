""""NBA gets no Price row" — the guarantee, and the test that keeps it.

The Price row in the shared explainer's template is the MARKET's quoted price: a
bookmaker's number, read against the model's. NBA facts never carry one, so the
row has nothing to draw and must not appear.

The naive version of this test — ``assert "slot" not in bundle`` — would pass
forever on a bundle that never had the key, and would keep passing if something
later added one. So this walks the WHOLE bundle, collects every ``slot`` value
anywhere in it, and asserts ``price`` is not among them. If anyone ever gives
NBA a quoted price slot, this fails.

Offline throughout: `_box_score_history` is stubbed, because a duel is not what
this file is about and reaching ESPN for one would be both slow and a real
network call from a test.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nba_predictor.api import facts as facts_mod


def _walk(value, path="bundle"):
    """Every dict in the payload, with the path that reached it."""
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from _walk(child, f"{path}[{i}]")


@pytest.fixture
def offline(monkeypatch):
    """Duels off, schedule doubled, store unreadable."""
    path = __import__("pathlib").Path("/nonexistent/x.sqlite")
    monkeypatch.setattr(facts_mod, "_box_score_history", lambda as_of: pd.DataFrame())
    monkeypatch.setattr(facts_mod, "_db_path", lambda: path)
    # `_market_line` reads the odds feed through the store; an NBA game with no
    # quoted line is exactly the case under test, so the odds rows are empty.
    monkeypatch.setattr(facts_mod, "_market_rows", lambda game_id: [])
    return monkeypatch


def _bundles(offline):
    """Both payload shapes /facts can return, upcoming and started."""
    from tests.test_facts import _game, _prediction_row, GAME_ID

    upcoming = {
        "game_id": GAME_ID, "game_date": "2026-01-16",
        "tip_off": "2026-01-16T00:30:00Z",
        "home_team": "BOS", "away_team": "MIA", "completed": False,
    }
    started = dict(upcoming, completed=True, home_pts=110, away_pts=100,
                   tip_off="2026-01-15T12:00:00Z")

    for label, game, rows in (("upcoming", upcoming, [_prediction_row()]),
                              ("started", started, [])):
        schedule = [game]
        yield label, {
            "sport": "nba",
            "id": str(game["game_id"]),
            "title": f"{game['away_team']} at {game['home_team']}",
            "starts_at": game["tip_off"],
            "status": "final" if game["completed"] else "upcoming",
            "pick_timing": "rebuilt",
            "pick": None,
            "markets": facts_mod._markets(game, rows[0] if rows else None),
            "drivers": [],
            "context": facts_mod._context(game, schedule, started=game["completed"], pick=None),
            "players": [],
            "record": None,
        }


def test_no_bundle_ever_yields_a_price_slot(offline):
    """Every `slot` anywhere in an NBA bundle, and `price` is not one of them."""
    for label, body in _bundles(offline):
        slots = [(path, node["slot"]) for path, node in _walk(body) if "slot" in node]
        assert not [s for s in slots if s[1] == "price"], (
            f"the {label} bundle claims a price slot at {slots}: NBA facts carry no "
            "quoted line, so there is nothing for a Price row to read it against"
        )


def test_no_nba_payload_quotes_a_bookmaker_price_for_a_game_outcome(offline):
    """The same guarantee, stated the way the template reads it: a Price row needs
    a quoted price, and an NBA market row must not carry one."""
    for label, body in _bundles(offline):
        for path, node in _walk(body):
            if "market" not in node:
                continue
            for key in ("price", "quoted_price", "book_price", "implied_probability"):
                assert key not in node, (
                    f"{label}{path} carries {key}={node[key]}: that is a quoted price, "
                    "which would put a Price row on an NBA page"
                )


def test_an_nba_context_never_carries_a_price_flavoured_key(offline):
    """And the context itself, which is where the duels land."""
    from tests.test_facts import _game

    game = _game()
    context = facts_mod._context(game, [game], started=False, pick={"label": "BOS", "prob": 0.62})
    for key in context:
        assert key in {"rest", "back_to_back", "matchups"}, (
            f"context carries {key!r}: an unexpected key on an NBA facts bundle"
        )
