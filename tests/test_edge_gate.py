"""The 5% edge gate (spec section 8, gap G5).

The repo computes and stores an edge on every market row and renders it, but
nothing ever *selects*. A 9% edge and a 1% edge are the same kind of object:
both are displayed, neither is ever a recommendation. This is the gate that turns
one into the other, and it carries the sibling contract (NFL `build_value_bet_table`
with `edge_threshold=0.05`, PL's 5% x data-confidence):

* **5% minimum edge.** Below it, the edge is inside the noise of a model whose
  own calibration is measured, not assumed.
* **At most one single per game, highest edge wins.** Two flagged singles on one
  game is two opinions on one event, and the loser is by construction the worse
  one.
* **No parlays.** Correlation between legs is not modelled here, so a parlay
  edge is not an edge.
* **Odds no older than one hour, else nothing at all.** A line recorded after
  tip-off is not a bettable price. This is the rule the whole loop exists to
  protect (spec section 14.3), and a stale slate must produce ZERO picks rather
  than a filtered subset -- a stale pick that survives is worse than no pick,
  because it looks actionable.
* **Same book, same point, both sides.** An edge computed between one book's
  total and another book's total is a difference between two shops' numbers, not
  a disagreement with the market. This was the NFL post-review fix.

Nothing here places a bet, publishes, or posts. It selects rows.
"""

from datetime import datetime, timedelta, timezone

import pytest

from nba_predictor.odds.edge_gate import (
    EDGE_THRESHOLD,
    MAX_ODDS_AGE,
    gated_picks,
    pair_total_sides,
)

NOW = datetime(2026, 10, 21, 18, 0, tzinfo=timezone.utc)


def row(
    *,
    game_id="g1",
    market="spread",
    selection="home",
    book="bovita",
    odds=-110,
    point=2.5,
    edge=0.06,
    minutes_old=5,
    model_id="m1",
):
    """One stored market row, shaped like `game_market_predictions`."""
    return {
        "game_id": game_id,
        "market": market,
        "selection": selection,
        "bookmaker": book,
        "american_odds": odds,
        "point": point,
        "edge": edge,
        "model_probability": 0.5 + edge,
        "market_probability": 0.5,
        "created_at": (NOW - timedelta(minutes=minutes_old)).isoformat(),
        "id": model_id,
    }


def market_pair(*, game_id="g1", market="spread", book="bovita", point=2.5,
                edge=0.06, minutes_old=5, other_edge=0.01):
    """A genuine two-sided market: one selection clearing the bar, one not.

    `refresh_odds` only ever writes a market as a pair -- it skips any book
    whose row count is not 2 -- so the gate's inputs are pairs. A lone row is
    half a market, and half a market has no edge in it.
    """
    home_side, away_side = ("over", "under") if market == "total" else ("home", "away")
    return [
        row(game_id=game_id, market=market, selection=home_side, book=book,
            point=point, edge=edge, minutes_old=minutes_old, model_id="m1"),
        row(game_id=game_id, market=market, selection=away_side, book=book,
            point=point, edge=other_edge, minutes_old=minutes_old, model_id="m2"),
    ]


def test_a_fresh_five_percent_edge_is_flagged():
    picks = gated_picks(market_pair(edge=0.06), now=NOW)
    assert [p["id"] for p in picks] == ["m1"]


def test_an_edge_under_five_percent_is_not_flagged():
    assert gated_picks(market_pair(edge=0.049), now=NOW) == []
    assert EDGE_THRESHOLD == 0.05


def test_the_threshold_is_inclusive_at_exactly_five_percent():
    """At the boundary either answer is defensible; what is not defensible is a
    gate whose behaviour at the boundary depends on floating point. Pinned."""
    assert gated_picks(market_pair(edge=0.05), now=NOW) != []


def test_stale_odds_produce_nothing_at_all():
    """Not a filtered subset -- zero.

    A stale row that still clears the edge bar is the dangerous case: it looks
    actionable and is not. The 9% edge below is deliberately far above the
    threshold, so only freshness can reject it.
    """
    stale = market_pair(edge=0.09, minutes_old=61)
    assert gated_picks(stale, now=NOW) == []
    assert MAX_ODDS_AGE == timedelta(hours=1)


def test_freshness_boundary_is_inclusive_at_exactly_one_hour():
    assert gated_picks(market_pair(edge=0.09, minutes_old=60), now=NOW) != []


def test_at_most_one_single_per_game_and_it_is_the_highest_edge():
    rows = [
        row(game_id="g1", selection="home", edge=0.06, model_id="six"),
        row(game_id="g1", selection="home", edge=0.11, model_id="eleven"),
        row(game_id="g1", selection="home", edge=0.08, model_id="eight"),
        row(game_id="g1", selection="away", edge=0.01, model_id="away"),
    ]
    picks = gated_picks(rows, now=NOW)
    assert [p["id"] for p in picks] == ["eleven"]
    assert len(picks) == 1, "one game produced more than one single"


def test_the_winner_is_chosen_across_books_too():
    """The highest edge on a game wins even when it is a different market or a
    different book -- the one-single rule is per game, not per book."""
    rows = [
        row(game_id="g1", market="spread", selection="home", book="bovita", edge=0.06, model_id="a"),
        row(game_id="g1", market="spread", selection="away", book="bovita", edge=0.01, model_id="a2"),
        row(game_id="g1", market="h2h", selection="away", book="draftkings", point=None, edge=0.09, model_id="b"),
    ]
    assert [p["id"] for p in gated_picks(rows, now=NOW)] == ["b"]


def test_different_games_are_each_flagged():
    rows = [
        row(game_id="g1", selection="home", edge=0.06, model_id="a"),
        row(game_id="g1", selection="away", edge=0.01, model_id="a2"),
        row(game_id="g2", selection="home", edge=0.07, model_id="b"),
        row(game_id="g2", selection="away", edge=0.01, model_id="b2"),
    ]
    assert sorted(p["id"] for p in gated_picks(rows, now=NOW)) == ["a", "b"]


def test_a_market_without_a_real_line_is_never_flagged():
    """No line, no cover probability, so no edge to grade. CFB's bug was
    fabricating a 0.5 spread to cover against; inventing a line here would be the
    same fabrication one level up."""
    assert gated_picks(market_pair(market="spread", point=None, edge=0.20), now=NOW) == []
    assert gated_picks(market_pair(market="total", point=None, edge=0.20), now=NOW) == []


def test_parlay_rows_are_never_flagged():
    """A parlay's legs are correlated and this repo does not model that, so a
    parlay edge is not an edge."""
    parlay = row(market="parlay", edge=0.40, model_id="parlay")
    assert gated_picks([parlay], now=NOW) == []


def test_totals_are_paired_same_book_same_point():
    """NFL's post-review fix.

    `over` at -110 from one book and `under` at -105 from another is not a
    market disagreeing with the model, it is two shops pricing differently. The
    pairing must refuse to produce an edge across books or across points.
    """
    same = [
        row(market="total", selection="over", book="bovita", point=220.5, edge=0.07),
        row(market="total", selection="under", book="bovita", point=220.5, edge=0.02),
    ]
    assert pair_total_sides(same) is not None, "a genuine same-book pair was refused"

    mixed_books = [
        row(market="total", selection="over", book="bovita", point=220.5),
        row(market="total", selection="under", book="draftkings", point=220.5),
    ]
    assert pair_total_sides(mixed_books) is None

    mixed_points = [
        row(market="total", selection="over", book="bovita", point=220.5),
        row(market="total", selection="under", book="bovita", point=221.5),
    ]
    assert pair_total_sides(mixed_points) is None


def test_a_total_row_with_no_counterpart_side_is_not_flagged():
    """One side of a total on its own is not a market. Flagging it would compare
    the model against half a market and call the difference an edge."""
    lonely = [row(market="total", selection="over", edge=0.09)]
    assert gated_picks(lonely, now=NOW) == []


def test_a_malformed_or_unparseable_timestamp_is_treated_as_stale():
    """An unreadable stamp must not read as fresh. Guessing 'now' would flag the
    oldest possible line."""
    bad = market_pair(edge=0.20)
    for r in bad:
        r["created_at"] = "not a timestamp"
    assert gated_picks(bad, now=NOW) == []


def test_a_pick_carries_the_numbers_a_reader_needs_to_check_it():
    """A flagged pick with no book, line or odds is a claim, not evidence."""
    picks = gated_picks(market_pair(edge=0.08), now=NOW)
    assert picks
    for key in ("game_id", "market", "selection", "bookmaker", "american_odds",
                "point", "edge", "model_probability", "market_probability",
                "created_at", "id"):
        assert key in picks[0], f"flagged pick is missing {key}"


def test_no_rows_no_picks():
    assert gated_picks([], now=NOW) == []

# --------------------------------------------------------------------------
# The API surface: the gate has to be reachable, and honest about being empty.
# --------------------------------------------------------------------------

def _seed_market_rows(db_path, *, minutes_old: int = 2):
    """Seed market rows stamped relative to the REAL clock.

    The gate-level tests inject `now=NOW`, but the endpoint takes no clock -- it
    uses the wall clock, as it must. So rows seeded against the module-level
    NOW constant are future-dated by the time the endpoint reads them, and
    "stale" fixtures silently become the freshest rows in the table.
    """
    from nba_predictor.tracking import store

    store.init_db(db_path)
    real_now = datetime.now(timezone.utc)
    for r in market_pair(edge=0.08, minutes_old=minutes_old):
        r["created_at"] = (real_now - timedelta(minutes=minutes_old)).isoformat()
        store.insert_market_prediction(
            db_path, game_id=r["game_id"], market=r["market"], selection=r["selection"],
            model_probability=r["model_probability"], market_probability=r["market_probability"],
            edge=r["edge"], bookmaker=r["bookmaker"], american_odds=r["american_odds"],
            point=r["point"], created_at=r["created_at"],
        )


def test_value_picks_endpoint_returns_only_gated_rows(tmp_path, monkeypatch):
    import json

    from fastapi.testclient import TestClient

    from nba_predictor.api import deps
    from nba_predictor.api.app import app

    db_path = tmp_path / "tracking.db"
    _seed_market_rows(db_path)
    # Keyed on the imported function object. `Depends(get_db_path)` captured it
    # at import, so rebinding the module attribute would key the override on a
    # different object and it would silently never match -- which reads as "no
    # qualifying picks" rather than as a wiring mistake.
    app.dependency_overrides[deps.get_db_path] = lambda: db_path

    response = TestClient(app).get("/value-picks")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["picks"]) == 1
    pick = payload["picks"][0]
    assert pick["edge"] >= EDGE_THRESHOLD
    assert pick["bookmaker"] and pick["market"]
    # The thresholds travel with the payload so a reader can see what was applied
    # rather than having to trust it.
    assert payload["edge_threshold"] == EDGE_THRESHOLD
    assert payload["max_odds_age_minutes"] == 60
    assert "not evidence of a profitable strategy" in payload["disclaimer"]
    json.dumps(payload)


def test_value_picks_is_empty_not_absent_when_nothing_qualifies(tmp_path):
    """No qualifying row is a real, common state -- a stale slate, or a day with
    no edge. It must read as an empty list with the thresholds still visible, not
    as a missing key or an error.

    Seeded deliberately stale: a 9%-edge row two hours old must produce nothing
    at all, not a filtered subset."""
    from fastapi.testclient import TestClient

    from nba_predictor.api import deps
    from nba_predictor.api.app import app

    db_path = tmp_path / "tracking.db"
    _seed_market_rows(db_path, minutes_old=120)
    # Keyed on the imported function object. `Depends(get_db_path)` captured it
    # at import, so rebinding the module attribute would key the override on a
    # different object and it would silently never match -- which reads as "no
    # qualifying picks" rather than as a wiring mistake.
    app.dependency_overrides[deps.get_db_path] = lambda: db_path

    payload = TestClient(app).get("/value-picks").json()
    assert payload["picks"] == []
    assert payload["n_odds_rows_considered"] == 0, (
        "a stale row reached the gate at all -- freshness is checked before the "
        "edge, so a 2-hour-old 9% edge must never be considered"
    )
    assert payload["edge_threshold"] == EDGE_THRESHOLD


def test_the_reader_really_excludes_rows_older_than_since(tmp_path):
    """The freshness filter, pinned where it actually happens.

    This exists because it was silently a no-op: `created_at` is TEXT holding
    an ISO-8601 string with a "T", and passing a `datetime` object into the SQL
    comparison made sqlite3 render it with a SPACE instead. " " sorts before
    "T", so *every* row looked newer than `since` and a two-hour-old line read
    as fresh -- which is precisely the condition the gate exists to catch.
    """
    from nba_predictor.tracking import store
    from nba_predictor.tracking.store import get_recent_market_predictions

    db_path = tmp_path / "tracking.db"
    store.init_db(db_path)
    real_now = datetime.now(timezone.utc)
    for minutes_old in (2, 120, 600):
        for r in market_pair(edge=0.08, minutes_old=minutes_old):
            store.insert_market_prediction(
                db_path, game_id=f"g{minutes_old}", market=r["market"],
                selection=r["selection"], model_probability=r["model_probability"],
                market_probability=r["market_probability"], edge=r["edge"],
                bookmaker=r["bookmaker"], american_odds=r["american_odds"], point=r["point"],
                created_at=(real_now - timedelta(minutes=minutes_old)).isoformat(),
            )

    since = (real_now - timedelta(minutes=60)).isoformat()
    rows = get_recent_market_predictions(db_path, since=since)

    assert rows, "the fresh rows were dropped"
    games = {r["game_id"] for r in rows}
    assert games == {"g2"}, (
        f"the reader returned {games}; only the 2-minute-old rows should be in "
        "range of a 60-minute window"
    )


def test_a_total_is_not_paired_with_another_games_total():
    """`/value-picks` hands the gate every recent row at once, so a counterpart
    has to be looked for inside the row's OWN game.

    Without that scoping a total for g1 paired with an unmatched total from g2:
    a valid row could be rejected, or -- worse -- an edge computed across two
    games.
    """
    g1 = market_pair(game_id="g1", market="total", edge=0.09)
    # g2 contributes a lone 'over' at the same point and book: it must not
    # become g1's counterpart.
    g2_lonely = [row(game_id="g2", market="total", selection="over",
                     book="bovita", point=220.5, edge=0.01)]

    picks = gated_picks(g1 + g2_lonely, now=NOW)
    assert [p["game_id"] for p in picks] == ["g1"], (
        "g1 has a complete same-book, same-point pair and must still be flagged; "
        "g2's lone over must not borrow g1's under"
    )

    # And with each game's own pair present, both are judged on their own merits.
    g2 = market_pair(game_id="g2", market="total", edge=0.07)
    picks = gated_picks(g1 + g2, now=NOW)
    assert sorted(p["game_id"] for p in picks) == ["g1", "g2"]


def test_a_spread_pairs_only_with_its_own_game_and_book():
    rows = [
        row(game_id="g1", selection="home", book="bovita", point=2.5, edge=0.09),
        # Different game, opposite side, matching book and point.
        row(game_id="g2", selection="away", book="bovita", point=2.5, edge=0.01),
    ]
    assert gated_picks(rows, now=NOW) == [], (
        "a spread was paired with the other side of a DIFFERENT game"
    )
