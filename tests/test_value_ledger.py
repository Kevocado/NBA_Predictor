"""The graded value ledger: snapshot, settle, closing line, CLV, weekly report.

Spec section 9. The gate produces picks; nothing records what happened to them.
Without a ledger the gate is an opinion generator, and there is no way to tell
later whether a positive-looking record was skill or a handful of lucky
outcomes.

Four properties, each a way this could quietly lie:

1. **Snapshots are immutable and strictly pre-tip.** A pick snapshotted after
   tip-off is not a bettable price, and one that can be overwritten is not a
   record. `tracking.timing.made_before_tip` already encodes the pre-tip rule
   and is reused rather than restated.

2. **Every flagged pick is settled, including losses and pushes.** A ledger
   that grades only hits is a highlight reel, not a ledger.

3. **CLV is measured against the closing line**, per pick. Closing line value is
   the one number that says whether the price was genuinely better than the
   market's final opinion -- a pick can win and still have been a bad price.

4. **The report shows yield whatever it is**, under copy that says plainly that
   this is not evidence of a profitable strategy. PL's framing, deliberately.

Nothing here places a bet.
"""

import pytest

from nba_predictor.tracking import store
from nba_predictor.tracking.value_ledger import (
    clv_pct,
    record_closing_line,
    settle_pick,
    snapshot_pick,
    weekly_ledger_report,
)

# A game tipped off 2026-10-21T19:00Z.
GAME = {"game_id": "g1", "home_team": "BOS", "away_team": "MIA", "tip_off": "2026-10-21T19:00:00+00:00"}
PRE_TIP = "2026-10-21T18:00:00+00:00"
POST_TIP = "2026-10-21T20:00:00+00:00"


def _db(tmp_path):
    path = tmp_path / "tracking.db"
    store.init_db(path)
    return str(path)


def _snap(db, **kw):
    args = dict(
        db_path=db, game_id="g1", market="spread", selection="home",
        bookmaker="bovita", american_odds=-110, point=2.5,
        model_probability=0.58, market_probability=0.5, edge=0.08,
        snapshot_at=PRE_TIP, game=GAME,
    )
    args.update(kw)
    return snapshot_pick(**args)


def test_snapshot_then_read_back(tmp_path):
    db = _db(tmp_path)
    _snap(db)
    from nba_predictor.tracking.value_ledger import read_ledger
    rows = read_ledger(db)
    assert len(rows) == 1
    assert rows[0]["edge"] == 0.08
    assert rows[0]["bookmaker"] == "bovita"
    assert rows[0]["outcome"] is None
    assert rows[0]["clv_pct"] is None


def test_a_post_tip_snapshot_is_refused(tmp_path):
    """A line recorded after tip-off is not a bettable price (spec 14.3)."""
    from nba_predictor.tracking.value_ledger import PostTipSnapshot
    db = _db(tmp_path)
    with pytest.raises(PostTipSnapshot):
        _snap(db, snapshot_at=POST_TIP)


def test_a_second_snapshot_does_not_displace_the_first(tmp_path):
    db = _db(tmp_path)
    _snap(db, model_probability=0.58, edge=0.08)
    _snap(db, model_probability=0.95, edge=0.45, snapshot_at="2026-10-21T18:30:00+00:00")
    from nba_predictor.tracking.value_ledger import read_ledger
    rows = read_ledger(db)
    assert len(rows) == 1, "a rerun filed a second pick for the same key"
    assert rows[0]["edge"] == 0.08, "the earliest snapshot was displaced"


def test_distinct_markets_and_books_are_distinct_picks(tmp_path):
    db = _db(tmp_path)
    _snap(db, market="spread", bookmaker="bovita")
    _snap(db, market="total", bookmaker="bovita", point=220.5)
    _snap(db, market="spread", bookmaker="draftkings")
    from nba_predictor.tracking.value_ledger import read_ledger
    assert len(read_ledger(db)) == 3


def test_a_won_pick_is_graded_with_its_error_and_clv(tmp_path):
    db = _db(tmp_path)
    _snap(db, selection="home", point=2.5, model_probability=0.58, edge=0.08)
    settle_pick(db, game_id="g1", market="spread", selection="home", bookmaker="bovita",
                outcome="won", settled_at="2026-10-22T02:00:00+00:00")

    from nba_predictor.tracking.value_ledger import read_ledger
    row = read_ledger(db)[0]
    assert row["outcome"] == "won"
    assert row["hit"] is True
    # The Brier term: a won pick is worth 0, and a lost pick is wrong by
    # (1 - model_probability) -- so a confident loser costs more than a narrow
    # one. A flat 0/1 would just restate `hit`.
    assert row["error"] == pytest.approx(0.0, abs=1e-9)


def test_a_lost_pick_is_recorded_not_dropped(tmp_path):
    """A ledger that grades only hits is a highlight reel."""
    db = _db(tmp_path)
    _snap(db)
    settle_pick(db, game_id="g1", market="spread", selection="home", bookmaker="bovita",
                outcome="lost", settled_at="2026-10-22T02:00:00+00:00")
    from nba_predictor.tracking.value_ledger import read_ledger
    row = read_ledger(db)[0]
    assert row["hit"] is False
    assert row["error"] == pytest.approx(1.0 - 0.58, abs=1e-6)


def test_an_unsettled_pick_is_never_counted_as_a_miss(tmp_path):
    """Same rule the game-level track record follows: ungraded is None, not
    False. A pick with no outcome has not been measured."""
    db = _db(tmp_path)
    _snap(db)
    from nba_predictor.tracking.value_ledger import read_ledger
    assert read_ledger(db)[0]["hit"] is None


def test_clv_is_measured_against_the_closing_line(tmp_path):
    """Closing line value: did the price beat the market's final opinion?"""
    db = _db(tmp_path)
    _snap(db, american_odds=-110)
    record_closing_line(db, game_id="g1", market="spread", selection="home",
                        bookmaker="bovita", closing_american_odds=-105,
                        recorded_at="2026-10-21T18:55:00+00:00")
    from nba_predictor.tracking.value_ledger import read_ledger
    row = read_ledger(db)[0]
    assert row["closing_american_odds"] == -105
    # Entered at -110, market closed at -105: we got a better price.
    assert row["clv_pct"] > 0


def test_clv_pct_sign_and_direction():
    """Entered at -110 with the market closing at -105 is a better price, so CLV
    is positive; the reverse is negative; the same price is zero.

    Deliberately NOT asserting antisymmetry. CLV here is `(close - entry) /
    close` -- the return measured against the closing price -- so the two
    directions are not mirror images: the denominators differ. Only the sign and
    the zero are properties worth pinning.
    """
    assert clv_pct(-110, -105) > 0, "a better entry price must read as positive CLV"
    assert clv_pct(-105, -110) < 0, "a worse entry price must read as negative CLV"
    assert clv_pct(-110, -110) == pytest.approx(0.0, abs=1e-9)


def test_clv_is_none_without_a_closing_line():
    assert clv_pct(-110, None) is None


def test_weekly_report_renders_hits_yield_and_clv(tmp_path):
    db = _db(tmp_path)
    _snap(db, game_id="g1", market="spread", selection="home", edge=0.08,
          model_probability=0.58, american_odds=-110)
    settle_pick(db, game_id="g1", market="spread", selection="home", bookmaker="bovita",
                outcome="won", settled_at="2026-10-22T02:00:00+00:00")
    record_closing_line(db, game_id="g1", market="spread", selection="home",
                        bookmaker="bovita", closing_american_odds=-105,
                        recorded_at="2026-10-21T18:55:00+00:00")

    report = weekly_ledger_report(db)
    assert "won" in report
    assert "CLV" in report
    assert "1" in report
    # The honest framing, not a profit claim.
    assert "not evidence of a profitable strategy" in report


def test_weekly_report_with_nothing_graded_says_so(tmp_path):
    db = _db(tmp_path)
    _snap(db)
    report = weekly_ledger_report(db)
    assert "0" in report
    assert "not evidence of a profitable strategy" in report
    assert "None" in report or "—" in report or "ungraded" in report.lower()


def test_settling_a_pick_that_was_never_snapshotted_records_nothing(tmp_path):
    """An outcome with no pick has nothing to grade; inventing a row would report
    a pick nobody made."""
    db = _db(tmp_path)
    settle_pick(db, game_id="gX", market="spread", selection="home", bookmaker="bovita",
                outcome="won", settled_at=POST_TIP)
    from nba_predictor.tracking.value_ledger import read_ledger
    assert read_ledger(db) == []


def test_settlement_is_immutable(tmp_path):
    """An outcome is a fact. A disagreeing second value is a contradiction, not
    an update."""
    db = _db(tmp_path)
    _snap(db)
    settle_pick(db, game_id="g1", market="spread", selection="home", bookmaker="bovita",
                outcome="won", settled_at="2026-10-22T02:00:00+00:00")
    with pytest.raises(store.ConflictingOutcome):
        settle_pick(db, game_id="g1", market="spread", selection="home", bookmaker="bovita",
                    outcome="lost", settled_at="2026-10-22T03:00:00+00:00")