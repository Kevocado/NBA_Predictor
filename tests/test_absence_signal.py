"""NBA's `absence` signal, and the trap this sport has in its strongest form.

Spec `2026-10-01-fixture-signals-design.md` §4 gives the example as *"Out: J.
Jacobs, our #2 rush projection (78 yds)"* — NFL, in YARDS. NBA's own units are a
POINTS TOTAl: `models/player_props.py` is an XGBRegressor returning a raw number,
and NBA serves no calibrated probability anywhere (the frontend's own header says
so, and cites that as why every row is `kind: "projection"`). So the row reads
*"Out: L. James, our #1 Points projection (31.5 pts)"* — a magnitude in the sport's
own units, which is what `figures.projection` means and why the shared component
draws a plain unsigned number.

## The trap, and why NBA is the worst case of the three sports

PL needed one fix to get an honest figure. NFL needs the same. **NBA needs the
figure to be *recovered*, because it was never sent.**

`api/routes.get_game_players` does this:

    out_ids = {entry["player_id"] for entry in resolve_out_players(injuries, ...)}
    for (player_id, stat), (pick, rebuilt) in picks.items():
        if player_id in out_ids:
            continue          # <-- the row is DROPPED, not flagged

and `OutPlayerOut`'s own docstring says the out player is *"Removed, not flagged in
place: no list, no bar, no rank position."* So on the wire an out player has no
`predicted_value`, no stat, and no rank at all — the frontend then re-applies the
removal defensively (`TopCalls.buildTopCalls`) precisely because it cannot trust the
API to have done it.

That is fine for a page that simply omits him, and it is the exact thing Phase 2
says is wrong: *"removing them promotes everyone below them, so #2 quietly becomes
#1."* An out player's absence is the largest single change to the fixture, and the
page shows it as nothing at all — while every player he outranks silently gains a
place.

So this adapter reads `picks_by_player_stat` — the FULL pool, out players included,
because the route filters *after* it — and ranks from that. The rank stated is the
rank the player HELD.

## Two decisions that are not obvious

**DOUBTFUL IS NOT OUT, and stays a flag.** `availability.is_doubtful` matches
`day-to-day`, and `DOUBTFUL_STATUSES` is a separate frozenset from `OUT_STATUSES`.
Those players STAY in the ranking; the frontend attaches a note. A doubtful player
is not an absence, and this endpoint reads only the out feed — partly for the same
reason `resolve_doubtful_players`'s docstring gives: `/players/out` is a removal
instruction the shipped frontend applies to every row it is given.

**TIES ACROSS CATEGORIES GO TO THE FIRST CATEGORY A READER SEES.** NBA's figures are
NOT comparable across stats — 31.5 points and 2.5 threes are the same number and
different quantities. So no value-based tiebreak, which would be a fabricated
weighting. The tiebreak is `STAT_TARGETS`' own order, which is the order the page
lists the categories in. Ranking by it invents nothing; ranking by magnitude would.

Run: python -m pytest tests/test_absence_signal.py -q
"""
from __future__ import annotations

import pytest

from nba_predictor.models.player_props import STAT_TARGETS
from nba_predictor.signals.absence import absence_signal, headline


def pick(value: float) -> dict:
    """One `(pick, rebuilt)` pair, the shape `picks_by_player_stat` yields."""
    return {"pick": {"predicted_value": value}, "rebuilt": False}


def out(player_id: str, name: str, status: str = "out") -> dict:
    return {
        "player_id": player_id,
        "player_name": name,
        "team": "LAL",
        "status": status,
        "source": "ESPN injury report",
        "dated": "2026-10-04",
    }


NAMES = {"p1": "Small", "p2": "Big", "p3": "Mid"}

# Three players on POINTS, and the model's best one is OUT. So removing him
# promotes the other two, and the honest rank for him is #1 -- not the #1 the
# filtered list would hand the survivor.
# A real rotation rather than one player per category: an out player who is the
# ONLY entry on a stat is trivially #1 there, which would make the rank tests pass
# for the wrong reason. `p1` is #3 on BOTH of his stats, so "the best rank he held"
# is #3 and the rank assertions below mean something.
POOL = {
    ("p2", "points"): pick(31.5),
    ("p3", "points"): pick(24.0),
    ("p1", "points"): pick(12.0),
    ("p3", "threes"): pick(3.0),
    ("p2", "threes"): pick(2.0),
    ("p1", "threes"): pick(0.5),
}


def test_names_the_out_player_and_states_the_figure_it_draws():
    sig = absence_signal("401", POOL, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert sig["kind"] == "absence"
    assert sig["sport"] == "nba"
    assert sig["game_id"] == "401"
    assert sig["visual"] == "absence_strip"
    assert sig["headline"]["text"] == "Out: Big, our #1 Points projection (31.5 pts)"
    assert sig["headline"]["figures"]["projection"] == 31.5


def test_the_rank_is_the_one_the_player_held_not_the_one_the_filter_leaves():
    """THE test. `p2` is out at 31.5 and tops POINTS. Rank the filtered pool and
    he does not appear at all; rank it with him in and he is #1. Every other rank
    on the page shifts up by one if he is removed, which is the defect."""
    sig = absence_signal("401", POOL, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert "#1" in sig["headline"]["text"]


def test_a_lower_ranked_out_player_states_its_own_rank():
    """Not always #1: an out player at 12.0 on a pool where a healthy player has
    31.5 is #3, and saying "#1" because an absence is news would be a rank that
    is simply false."""
    sig = absence_signal(
        "401", POOL, [out("p1", "Small")], NAMES
    )
    assert sig is not None
    assert "#3" in sig["headline"]["text"]
    assert "12 pts" in sig["headline"]["text"]


def test_a_doubtful_player_is_not_an_absence():
    """`availability.DOUBTFUL_STATUSES` is a separate frozenset and those players
    STAY in the ranking. A day-to-day player is not out."""
    sig = absence_signal(
        "401", POOL, [out("p2", "Big", status="day-to-day")], NAMES
    )
    assert sig is None


def test_a_status_that_is_neither_out_nor_doubtful_is_not_an_absence():
    sig = absence_signal("401", POOL, [out("p2", "Big", status="active")], NAMES)
    assert sig is None


def test_no_out_players_means_no_signal():
    assert absence_signal("401", POOL, [], NAMES) is None


def test_an_out_player_with_no_projection_yields_no_signal():
    """He is in the injury report and absent from `picks`. There is no figure to
    draw and no rank to state, so there is no row -- an empty-state card is the
    alternative and it says nothing."""
    assert absence_signal("401", POOL, [out("nobody", "Ghost")], NAMES) is None


def test_an_empty_pool_yields_no_signal():
    assert absence_signal("401", {}, [out("p2", "Big")], NAMES) is None


def test_the_better_rank_wins_across_categories():
    """One out player at #1 on threes and another at #2 on points: the #1 is the
    bigger absence, because the rank is the claim the row makes about its own
    category."""
    pool = {
        ("p2", "threes"): pick(3.1),
        ("p3", "threes"): pick(2.0),
        ("p3", "points"): pick(30.0),
        ("p1", "points"): pick(28.0),
    }
    sig = absence_signal(
        "401", pool, [out("p2", "Big"), out("p1", "Small")], NAMES
    )
    assert sig is not None
    assert "Big" in sig["headline"]["text"]
    assert "#1" in sig["headline"]["text"]


def test_a_rank_tie_goes_to_the_category_the_page_lists_first():
    """Not to the bigger number. 31.5 points and 2.5 threes are the same number
    and different quantities, so a magnitude tiebreak would be a fabricated
    weighting. `STAT_TARGETS`' order is the page's own display order."""
    # The two rules are made to DISAGREE, which is the only way this test can tell
    # them apart: the threes figure is the bigger number, so a magnitude tiebreak
    # picks Threes, while the page's display order picks Points. Written the natural
    # way round (points 31.5, threes 3.0) BOTH rules pick Points and the test would
    # pass for the wrong reason -- which is exactly what the first version did.
    pool = {
        ("p2", "threes"): pick(9.0),
        ("p2", "points"): pick(5.0),
    }
    sig = absence_signal("401", pool, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert "Points" in sig["headline"]["text"]
    # And the order it is asserting IS the page's order.
    assert STAT_TARGETS.index("points") < STAT_TARGETS.index("threes")
    # The larger figure was NOT chosen, which is the point.
    assert "Threes" not in sig["headline"]["text"]


def test_strength_rises_as_the_rank_improves():
    """Rank is the claim, so rank is what orders the row. Bounded and monotone, so
    two rows can never tie at an arbitrary ceiling."""
    first = absence_signal("401", POOL, [out("p2", "Big")], NAMES)
    third = absence_signal("401", POOL, [out("p1", "Small")], NAMES)
    assert first is not None and third is not None
    assert first["strength"] > third["strength"]
    assert 0.0 < third["strength"] <= 1.0 <= first["strength"]


def test_n_counts_players_not_graded_games():
    pool = dict(POOL)
    pool[("p3", "points")] = pick(24.0)
    sig = absence_signal(
        "401", pool, [out("p2", "Big"), out("p1", "Small")], NAMES
    )
    assert sig is not None
    # Two injured players. The spec's n >= 30 rate floor would refuse this row,
    # which is the category error the component's `absence_strip` exemption exists
    # to prevent.
    assert sig["n"] == 2


def test_a_whole_number_projection_states_no_decimal():
    """`fmt.stat` would print 31.5 fine but 30.0 as "30.0", and a 30-point
    projection with a tenth of a point on it is precision the model never had."""
    pool = {("p2", "points"): pick(30.0)}
    sig = absence_signal("401", pool, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert "30 pts" in sig["headline"]["text"]
    assert "30.0" not in sig["headline"]["text"]


def test_a_fractional_projection_keeps_its_one_decimal():
    pool = {("p2", "points"): pick(29.5)}
    sig = absence_signal("401", pool, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert "29.5 pts" in sig["headline"]["text"]


def test_a_missing_name_falls_back_to_the_player_id_rather_than_nothing():
    """The id IS in the payload. An unnamed row would be dropped, and the reader
    loses the single most consequential fact on the fixture over a lookup the
    payload can already answer."""
    sig = absence_signal("401", POOL, [out("p2", "Big")], {})
    assert sig is not None
    assert "p2" in sig["headline"]["text"]


def test_the_game_id_is_carried_as_a_string():
    sig = absence_signal(401, POOL, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert sig["game_id"] == "401"


def test_pre_kickoff_only_is_true():
    sig = absence_signal("401", POOL, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert sig["pre_kickoff_only"] is True


def test_the_source_is_the_injurys_own_provenance_not_a_restatement():
    """ESPN's own words and its own date. Not "injured", which would be this
    repo's word for a status the feed publishes."""
    sig = absence_signal("401", POOL, [out("p2", "Big")], NAMES)
    assert sig is not None
    assert "ESPN injury report" in sig["source"]
    assert "2026-10-04" in sig["source"]


def test_a_headline_is_within_the_wordspec_limit():
    """Spec §4 caps a headline at 12 words."""
    assert len(headline("J. Smith", "Points", 1, 31.5).split()) <= 12


def test_the_headline_uses_no_sportsbook_or_certainty_vocabulary():
    text = headline("L. James", "Points", 1, 31.5).lower()
    for word in ("lock", "guaranteed", "best bet", "edge", "value", "sure"):
        assert word not in text


def test_a_non_finite_projection_is_skipped_rather_than_stated():
    """`picks` comes out of a store; a NaN predicted_value must not become
    "nan pts" or a NaN rank that silently sorts first."""
    pool = {("p2", "points"): pick(float("nan")), ("p3", "points"): pick(10.0)}
    sig = absence_signal("401", pool, [out("p2", "Big")], NAMES)
    assert sig is None
