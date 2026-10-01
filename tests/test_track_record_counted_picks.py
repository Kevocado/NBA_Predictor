"""The track record counts every recorded pick, and says when each was made.

Kevin, 2026-10-01, in `predictor-hub`
`docs/superpowers/specs/2026-10-01-track-record-counts-every-pick.md` (merged as
predictor-hub #66), verbatim:

  "i dont really care about picks made after kickoff because im always re
   running the models ... with every model change it will stop tracking ...
   make it that whats recorded remains recorded and then just use every
   prediction we make for the track record stuff."

This reverses the rule NBA shipped with. Before it, `pre_tip_picks` took the
LATEST row made before tip-off and every game whose only rows came from the
retrain backtest was reported as `n_rebuilt` and left out of every rate. That
is the "with every model change it will stop tracking" failure measured: a
model re-run on already-played games contributed nothing, and the headline was
built only from picks captured before the lights came on.

What replaced it:

  1. Recorded stays recorded. Nothing overwrites or deletes a pick.
  2. One counted pick per key, the EARLIEST recorded. A later rerun is history:
     it neither replaces the counted pick nor counts a second time -- otherwise
     re-running until the model was right would be free.
  3. The headline counts every counted pick, whenever it was made. The figure
     beside it is the pre-tip SUBSET, with its own n.
  4. Honesty is disclosure, not exclusion: `made_before_tip` is DERIVED on
     every read from the pick's own `created_at` against the game's tip-off,
     compared as UTC instants, and fails closed to False. Never a stored flag,
     never a `true` the timestamps do not prove.

**The counting key differs per record type, and getting that wrong is
destructive.** Game markets key on `(game, market)`; player props key on
`(game, player_id, stat)`. Collapsing props to game level would delete the
player record -- `test_player_props_are_keyed_by_game_player_and_stat_not_by_
game_alone` pins two players on one stat in one game and requires both.

**`compute_model_calibration` is deliberately NOT relaxed** (see
tests/test_calibration_service.py): those buckets are a price the hub trades
against, not a track record.
"""
import json
import re
import sqlite3
from pathlib import Path

import pytest

from nba_predictor.services import hub_service
import nba_predictor.tracking.store as store_module
from nba_predictor.tracking import store
from nba_predictor.tracking.player_props import resolved_player_props
from nba_predictor.tracking.timing import made_before_tip

REPO = Path(__file__).resolve().parents[1]


def _completed(game_id, home, away, home_pts, away_pts, **over):
    return {
        "game_id": game_id, "game_date": "2026-03-01", "home_team": home, "away_team": away,
        "completed": True, "home_pts": home_pts, "away_pts": away_pts, **over,
    }


def _db(tmp_path):
    db = tmp_path / "tracking.db"
    store.init_db(db)
    return db


def _outcome(db, game_id, prob, at, *, version="v1"):
    return store.insert_prediction(
        db, game_id=game_id, created_at=at, model_version=version,
        home_win_prob=prob, predicted_margin=1.0, predicted_total=220.0,
    )


def _row(db, records, market="game_outcome"):
    return next(r for r in records if r.market == market)


# --- rule 2: the counted pick is the EARLIEST recorded one ---------------------

def test_the_counted_pick_is_the_earliest_recorded_one_not_a_rerun(tmp_path):
    """A rerun neither replaces the counted pick nor counts a second time.

    One game, TWO picks made before tip-off: the model changed between them.
    The counted pick is the FIRST (0.9, and the home team lost, so it is the
    wrong one); the second (0.1, which would have been right) is history.

    This is the discriminating case: the old rule took the LATEST pre-tip row,
    so it would have graded 0.1 and reported 1/1. The new rule reports 0/1, and
    a rule that counted both would report 1/2. The second pick here is
    deliberately the flattering one, so a double count is also the flattering
    error rather than an obvious one.
    """
    db = _db(tmp_path)
    _outcome(db, "g1", 0.9, "2026-03-01T09:00:00+00:00", version="v1")
    _outcome(db, "g1", 0.1, "2026-03-01T10:00:00+00:00", version="v2")
    schedule = [_completed("g1", "BOS", "MIA", 100, 110)]  # home lost

    games = _row(db, hub_service.compute_track_record(db, schedule), "game_outcome")

    assert (games.total_predictions, games.correct_predictions) == (1, 0)
    assert games.hit_rate == 0.0
    # The rerun is still in the table (rule 1) and is disclosed as not counted.
    assert [p.counted for p in games.per_pick] == [True, False]
    assert {p.created_at for p in games.per_pick} == {
        "2026-03-01T09:00:00+00:00", "2026-03-01T10:00:00+00:00",
    }


def test_market_markets_count_the_earliest_recorded_run_per_game(tmp_path):
    """The same rule on the priced markets, where a "run" is one odds refresh.

    `refresh_odds` writes both sides for every bookmaker on every run, so a
    game carries several rows per run and several runs. The counted pick is
    the earliest run's, and within it the side the model priced highest. The
    later run is the better one here (0.95 on the winner), so a grader that
    took the latest -- which is what `latest_pre_tip` did -- would report 1/1.
    """
    db = _db(tmp_path)
    for selection, prob, at in (("MIA", 0.55, "2026-03-01T09:00:00+00:00"),
                                ("BOS", 0.45, "2026-03-01T09:00:00+00:00")):
        store.insert_market_prediction(
            db, game_id="g1", market="h2h", selection=selection, model_probability=prob,
            market_probability=0.5, edge=0.05, bookmaker="DraftKings", american_odds=110,
            created_at=at,
        )
    for selection, prob, at in (("BOS", 0.95, "2026-03-01T10:00:00+00:00"),
                                ("MIA", 0.05, "2026-03-01T10:00:00+00:00")):
        store.insert_market_prediction(
            db, game_id="g1", market="h2h", selection=selection, model_probability=prob,
            market_probability=0.5, edge=0.45, bookmaker="DraftKings", american_odds=-200,
            created_at=at,
        )
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]  # home won

    h2h = _row(db, hub_service.compute_track_record(db, schedule), "h2h")

    assert (h2h.total_predictions, h2h.correct_predictions) == (1, 0), (
        "the counted pick is the latest run, not the earliest recorded one"
    )
    counted = [p for p in h2h.per_pick if p.counted]
    assert len(counted) == 1
    assert counted[0].created_at == "2026-03-01T09:00:00+00:00"
    assert counted[0].pick == "MIA"
    assert counted[0].actual == "BOS"


# --- rules 3 and 4: headline over all counted picks, pre-tip beside it --------

def test_the_headline_counts_a_pick_recorded_after_tip_off_with_the_pre_tip_subset_beside_it(tmp_path):
    """The swap, on the market the page leads with.

    Old rule: `test_track_record_counts_only_picks_made_before_tip_off` in
    tests/test_hub_service.py asserted `total_predictions == 1` and
    `n_rebuilt == 1` for exactly this fixture -- g1 a pre-tip pick it got right,
    g2 a backtest row it also got right, dropped. New: both count, 2/2, and
    the pre-tip subset beside them is the 1/1 the old headline showed.
    """
    db = _db(tmp_path)
    _outcome(db, "g1", 0.7, "2026-03-01T09:00:00+00:00")
    _outcome(db, "g2", 0.4, "2026-09-20T08:00:00+00:00", version="v2")
    schedule = [_completed("g1", "BOS", "MIA", 110, 100), _completed("g2", "LAL", "GSW", 95, 105)]

    games = _row(db, hub_service.compute_track_record(db, schedule), "game_outcome")

    assert (games.total_predictions, games.correct_predictions) == (2, 2)
    assert games.hit_rate == 1.0
    # The secondary figure: the pre-tip subset, with its own n.
    assert games.n_pre_tip == 1
    assert games.pre_tip.total_predictions == 1
    assert games.pre_tip.correct_predictions == 1
    assert games.pre_tip.hit_rate == 1.0
    # n_rebuilt keeps its name and is now the reconciliation between the two
    # figures, not an exclusion: counted picks made at or after their own tip.
    assert games.n_rebuilt == 1
    assert games.total_predictions == games.pre_tip.total_predictions + games.n_rebuilt
    # Disclosure is per pick, on the row, with the row's own timestamp.
    labels = {p.game_id: p.made_before_tip for p in games.per_pick}
    assert labels == {"g1": True, "g2": False}


def test_a_game_with_no_pre_tip_pick_at_all_is_counted_and_labelled_not_pre_tip(tmp_path):
    """The case the old rule erased: a game whose every pick came from a rerun.

    Under `pre_tip_picks` this game was `n_rebuilt` and contributed to no rate
    on the page. It is a recorded pick, so it counts -- and it can never be
    presented as a pre-game one.
    """
    db = _db(tmp_path)
    _outcome(db, "g1", 0.6, "2026-09-20T08:00:00+00:00", version="v2")
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]

    games = _row(db, hub_service.compute_track_record(db, schedule), "game_outcome")

    assert (games.total_predictions, games.correct_predictions) == (1, 1)
    assert games.n_pre_tip == 0
    assert games.pre_tip.total_predictions == 0
    assert games.pre_tip.hit_rate is None, "never a fabricated 0% over an empty subset"
    assert games.per_pick[0].made_before_tip is False


def test_the_weekly_table_sums_to_the_headline_and_to_the_pre_tip_subset(tmp_path):
    """Both identities hold, or the page shows two counts that disagree."""
    from datetime import date

    db = _db(tmp_path)
    _outcome(db, "g1", 0.7, "2026-03-02T09:00:00+00:00")
    _outcome(db, "g2", 0.4, "2026-03-16T08:00:00+00:00", version="v2")
    _outcome(db, "g3", 0.6, "2026-03-16T09:00:00+00:00", version="v2")
    schedule = [
        _completed("g1", "BOS", "MIA", 110, 100, game_date="2026-03-02"),
        _completed("g2", "LAL", "GSW", 95, 105, game_date="2026-03-16"),
        _completed("g3", "BOS", "MIA", 100, 110, game_date="2026-03-16"),
    ]

    games = _row(db, hub_service.compute_track_record(db, schedule, today=date(2026, 3, 20)),
                 "game_outcome")

    assert sum(w.n for w in games.weekly) == games.total_predictions
    assert sum(w.correct for w in games.weekly) == games.correct_predictions
    assert sum(w.n for w in games.pre_tip.weekly) == games.pre_tip.total_predictions
    assert sum(w.correct for w in games.pre_tip.weekly) == games.pre_tip.correct_predictions


def test_a_push_stays_out_of_both_rates_and_stays_in_the_reconciliation(tmp_path):
    """A pick that landed exactly on the line is not a miss, in either figure.

    `n_rebuilt` counts GRADED counted picks made at or after tip, so the
    identity `total == pre_tip.total + n_rebuilt` survives a push instead of
    quietly absorbing it.
    """
    db = _db(tmp_path)
    for game_id, at in (("g_pre", "2026-03-01T09:00:00+00:00"),
                        ("g_post", "2026-09-20T08:00:00+00:00")):
        store.insert_market_prediction(
            db, game_id=game_id, market="spread", selection="BOS", model_probability=0.6,
            market_probability=0.5, edge=0.1, bookmaker="DraftKings", american_odds=-110,
            created_at=at, point=-4.0,
        )
    schedule = [
        _completed("g_pre", "BOS", "MIA", 104, 100),   # margin 4 == the line: push
        _completed("g_post", "BOS", "MIA", 104, 100),
    ]

    spread = _row(db, hub_service.compute_track_record(db, schedule), "spread")

    assert spread.n_push == 2
    assert (spread.total_predictions, spread.correct_predictions) == (0, 0)
    assert spread.hit_rate is None
    assert spread.n_pre_tip == 0
    assert spread.total_predictions == spread.pre_tip.total_predictions + spread.n_rebuilt


# --- `made_before_tip` is derived from UTC instants, not wall clocks ----------

def test_made_before_tip_is_derived_from_utc_instants_not_wall_clock_strings():
    """Two straddles where comparing the strings gives the opposite answer.

    A `tip_off` handed over by the schedule cache can carry any offset, so the
    wall clocks on the two sides can read in one order while the instants read
    in the other. Each case asserts the STRING order first, so the test cannot
    quietly stop straddling: if the offsets were ever normalised upstream, the
    precondition fails and says so instead of the assertion passing for the
    wrong reason.

    Case A: `23:30:00+09:00` is 14:30Z against a `15:00:00+00:00` tip -- before.
            The strings read "23:30" against "15:00", which is after.
    Case B: `23:30:00-11:00` is 10:30Z against a `23:45:00+14:00` tip (09:45Z) --
            after. The strings read "23:30" against "23:45", which is before.
            +14:00 against -11:00 is the widest gap the zones allow, which is
            how this pair is forced rather than picked.
    """
    pre = {"game_id": "g1", "game_date": "2026-09-12", "tip_off": "2026-09-12T15:00:00+00:00"}
    post = {"game_id": "g2", "game_date": "2026-09-12", "tip_off": "2026-09-12T23:45:00+14:00"}

    # The wall-clock reading, stated as a precondition.
    assert "2026-09-12T23:30:00+09:00" > "2026-09-12T15:00:00+00:00"  # would say "after"
    assert "2026-09-12T23:30:00-11:00" < "2026-09-12T23:45:00+14:00"  # would say "before"

    assert made_before_tip("2026-09-12T23:30:00+09:00", pre) is True
    assert made_before_tip("2026-09-12T23:30:00-11:00", post) is False


def test_made_before_tip_ignores_the_machine_timezone():
    """The derivation reads the offsets in the data, never the host's clock.

    The same two instants under four host zones must agree; a derivation that
    dropped the offset and read the machine's zone would not.
    """
    import os
    import time

    stamp, game = "2026-09-12T23:30:00+09:00", {
        "game_id": "g1", "game_date": "2026-09-12", "tip_off": "2026-09-12T15:00:00+00:00",
    }
    original = os.environ.get("TZ")
    try:
        answers = [
            (os.environ.__setitem__("TZ", zone), time.tzset(), made_before_tip(stamp, game))[2]
            for zone in ("UTC", "America/Los_Angeles", "Asia/Tokyo", "Pacific/Kiritimati")
        ]
    finally:
        if original is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original
        time.tzset()

    assert answers == [True, True, True, True], answers
    # Equal instants are not "before", whichever way each side is written:
    # 2026-09-13T00:00+09:00 is 2026-09-12T15:00Z, the tip-off itself.
    assert made_before_tip("2026-09-12T15:00:00+00:00", game) is False
    assert made_before_tip("2026-09-13T00:00:00+09:00", game) is False
    assert made_before_tip("2026-09-12T15:00:01+00:00", game) is False


def test_a_pick_whose_timing_cannot_be_proven_counts_and_is_never_labelled_pre_tip(tmp_path):
    """Disclosure fails closed; exclusion no longer exists.

    Two ways the timestamps cannot prove anything: a `created_at` that does not
    parse, and a game the schedule has no `game_date` for at all. Both are
    recorded picks, so both count. Neither may be labelled pre-tip, because
    the rule is never to backfill a `true` the timestamps do not prove.
    """
    db = _db(tmp_path)
    _outcome(db, "g_bad", 0.6, "not a time", version="v2")
    _outcome(db, "g_undated", 0.6, "2026-03-01T09:00:00+00:00")
    schedule = [
        {"game_id": "g_bad", "game_date": "2026-03-01", "home_team": "BOS", "away_team": "MIA",
         "completed": True, "home_pts": 110, "away_pts": 100},
        {"game_id": "g_undated", "home_team": "BOS", "away_team": "MIA",
         "completed": True, "home_pts": 110, "away_pts": 100},
    ]

    games = _row(db, hub_service.compute_track_record(db, schedule), "game_outcome")

    assert (games.total_predictions, games.correct_predictions) == (2, 2)
    assert games.n_pre_tip == 0
    assert all(p.made_before_tip is False for p in games.per_pick)


def test_every_counted_pick_carries_its_own_timestamp_and_the_pre_tip_label(tmp_path):
    """Rule 4 asserted on the published rows, not the aggregate.

    A `per_pick` list carrying the label without the timestamp could not be
    audited by a reader, which is the whole point of disclosing per pick.
    """
    db = _db(tmp_path)
    _outcome(db, "g_pre", 0.7, "2026-03-01T09:00:00+00:00")
    _outcome(db, "g_post", 0.4, "2026-09-20T08:00:00+00:00", version="v2")
    schedule = [_completed("g_pre", "BOS", "MIA", 110, 100), _completed("g_post", "LAL", "GSW", 105, 95)]

    per_pick = _row(db, hub_service.compute_track_record(db, schedule), "game_outcome").per_pick

    assert {p.game_id for p in per_pick} == {"g_pre", "g_post"}
    for pick in per_pick:
        assert pick.created_at, "a pick row was published without its own timestamp"
        assert isinstance(pick.made_before_tip, bool)
        assert isinstance(pick.hit, (bool, type(None)))
        assert pick.pick and pick.actual, "a row a reader cannot check against the result"
    pre = next(p for p in per_pick if p.game_id == "g_pre")
    assert (pre.pick, pre.actual, pre.made_before_tip) == ("BOS", "BOS", True)


# --- the earliest-recorded helper, as a unit ---------------------------------

def test_earliest_recorded_keeps_one_row_per_key_and_sorts_by_utc_instant():
    """Rule 2 on rows that CAN hold two per key, which the tables can.

    from nba_predictor.tracking.timing import earliest_recorded

    `predictions`, `game_market_predictions` and `player_prediction_snapshots`
    have no uniqueness constraint, so a rerun genuinely lands twice; this is
    the read-side rule that stops it being graded twice, not a belt to braces.

    Imported here so this module still COLLECTS against `main`, where the rule
    does not exist yet: a collection error would report one failure for
    thirteen tests instead of thirteen failures.

    The winner is the earliest UTC INSTANT even when it is listed last, a
    second player on the same key-shape is a different pick, and a stamp that
    cannot be parsed sorts last so it never displaces a provable row.
    """
    from nba_predictor.tracking.timing import earliest_recorded

    rows = [
        {"game_id": "g1", "created_at": "2026-09-12T15:00:00+00:00", "n": 2.0},
        # Same instant as the row above, written as +09:00: a tie, first in wins.
        {"game_id": "g1", "created_at": "2026-09-13T00:00:00+09:00", "n": 3.0},
        # 14:30Z, EARLIER than both, but listed last: this is the counted pick.
        {"game_id": "g1", "created_at": "2026-09-12T23:30:00+09:00", "n": 1.0},
        {"game_id": "g2", "created_at": "2026-09-12T16:00:00+00:00", "n": 4.0},
    ]

    assert [r["n"] for r in earliest_recorded(rows)] == [1.0, 4.0]


def test_earliest_recorded_keeps_an_unparseable_stamp_only_when_nothing_provable():
    """Fail-closed ordering, and no silent loss of a real pick."""
    from nba_predictor.tracking.timing import earliest_recorded

    unparseable = [{"game_id": "g1", "created_at": "not a time", "n": 9.0}]
    assert [r["n"] for r in earliest_recorded(unparseable)] == [9.0], (
        "a pick exists; losing it would break rule 1"
    )
    mixed = unparseable + [{"game_id": "g1", "created_at": "2026-09-12T16:00:00+00:00", "n": 4.0}]
    assert [r["n"] for r in earliest_recorded(mixed)] == [4.0], (
        "a row that cannot be proven earliest must not displace one that can"
    )


# --- player props: keyed on (game, player, stat), not (game, stat) -----------

def test_player_props_are_keyed_by_game_player_and_stat_not_by_game_alone(tmp_path):
    """Two players, one stat, one game: both are picks and both must survive.

    Collapsing player props to the game's level -- the literal reading of "one
    counted pick per (game, market)" -- would delete one of these rows. A
    points pick is one pick per player per game, so the key carries
    `player_id`. Pinned because the collapse is silent: the MAE would still be
    a number, just the wrong one, computed over half the record.
    """
    db = _db(tmp_path)
    for player_id, predicted, actual in (("p1", 20.0, 24.0), ("p2", 30.0, 27.0)):
        store.insert_player_prediction(
            db, game_id="g1", player_id=player_id, stat="points",
            predicted_value=predicted, created_at="2026-03-01T09:00:00+00:00",
        )
        store.insert_player_outcome(
            db, game_id="g1", player_id=player_id, stat="points",
            actual_value=actual, recorded_at="2026-03-02T09:00:00+00:00",
        )
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]

    rows = resolved_player_props(db, schedule)

    assert sorted(r["player_id"] for r in rows) == ["p1", "p2"], (
        "collapsing player props to the game deleted a player's record"
    )


def test_the_props_mae_counts_a_pick_recorded_after_tip_off_with_the_pre_tip_mae_beside_it(tmp_path):
    """The swap on the props MAE (points/rebounds/assists/threes).

    Old rule: `resolved_player_props` skipped every rebuilt row, so a game whose
    only points pick came from a rerun contributed nothing -- and with it the
    "no error estimate yet" branch on the page. New: it counts, `mae` is the
    headline over every counted pick, and `mae_pre_tip` is the honest read of
    what the model would have said on the night, over its own subset.
    """
    db = _db(tmp_path)
    store.insert_player_prediction(
        db, game_id="g_pre", player_id="p1", stat="points",
        predicted_value=20.0, created_at="2026-03-01T09:00:00+00:00",
    )
    store.insert_player_outcome(
        db, game_id="g_pre", player_id="p1", stat="points",
        actual_value=24.0, recorded_at="2026-03-02T09:00:00+00:00",
    )
    store.insert_player_prediction(
        db, game_id="g_post", player_id="p2", stat="points",
        predicted_value=30.0, created_at="2026-09-20T08:00:00+00:00",
    )
    store.insert_player_outcome(
        db, game_id="g_post", player_id="p2", stat="points",
        actual_value=48.0, recorded_at="2026-09-21T08:00:00+00:00",
    )
    schedule = [_completed("g_pre", "BOS", "MIA", 110, 100),
                _completed("g_post", "LAL", "GSW", 105, 95)]

    from nba_predictor.models.player_props import in_sample_mae_by_stat

    rows = resolved_player_props(db, schedule)
    assert [r["made_before_tip"] for r in sorted(rows, key=lambda r: r["player_id"])] == [True, False]
    # The headline counts both: (|20-24| + |30-48|) / 2 = 11.0
    assert in_sample_mae_by_stat(rows)["points"] == pytest.approx(11.0)
    # The secondary figure is the pre-tip subset alone.
    pre = [r for r in rows if r["made_before_tip"]]
    assert in_sample_mae_by_stat(pre)["points"] == pytest.approx(4.0)


def test_the_props_counted_pick_is_the_earliest_recorded_one_per_player(tmp_path):
    """A rerun of one player's prop neither replaces it nor counts twice."""
    db = _db(tmp_path)
    store.insert_player_prediction(
        db, game_id="g1", player_id="p1", stat="points",
        predicted_value=20.0, created_at="2026-03-01T09:00:00+00:00",
    )
    store.insert_player_prediction(
        db, game_id="g1", player_id="p1", stat="points",
        predicted_value=21.0, created_at="2026-03-01T10:00:00+00:00",
    )
    store.insert_player_outcome(
        db, game_id="g1", player_id="p1", stat="points",
        actual_value=24.0, recorded_at="2026-03-02T09:00:00+00:00",
    )
    schedule = [_completed("g1", "BOS", "MIA", 110, 100)]

    rows = resolved_player_props(db, schedule)

    assert [(r["predicted_value"], r["actual_value"]) for r in rows] == [(20.0, 24.0)]


# --- the read side sees each row's own provenance ----------------------------

def test_the_grade_writes_never_touch_the_stamp_both_rules_read(tmp_path):
    """The load-bearing invariant, and it is now load-bearing for MORE.

    Two things the record promises are read off `created_at`: which pick counts
    (the earliest recorded one) and whether it is labelled `made_before_tip`.
    A write path able to rewrite that column could put a rerun in the record as
    the earliest pick, and could backfill a `true` the timestamps do not prove --
    which is the one thing the 2026-10-01 decision still forbids outright.

    NBA's structure makes that impossible, and this asserts the structure
    rather than trusting it:

      * `tracking/store.py` writes with INSERT only -- no UPDATE and no DELETE
        anywhere in it -- so no writer (the ingest backtest, the odds poller,
        the admin route) can restamp a row after the fact. Outcomes are graded
        by inserting into `game_player_outcomes`, a different table, which is
        what makes this hold.
      * no prediction table declares a timing column, so `made_before_tip` has
        nowhere to be stored and must be derived on every read.

    Add an UPDATE that sets `created_at`, or a `made_before_tip` column, and
    this fails -- which is the point of writing it down.
    """
    source = Path(store_module.__file__).read_text()
    code = re.sub(r'""".*?"""', "", source, flags=re.S)  # docstrings are not code

    assert not re.search(r"\bUPDATE\b", code, re.I), (
        "tracking/store.py now writes an UPDATE; if it can restamp created_at, both "
        "the earliest-pick rule and the pre-tip label become corruptible"
    )
    assert not re.search(r"\bDELETE\b", code, re.I), (
        "tracking/store.py now deletes rows; rule 1 is that recorded stays recorded"
    )
    for table in ("predictions", "game_market_predictions", "player_prediction_snapshots"):
        declared = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\)", store_module.SCHEMA, re.S)
        assert declared, f"{table} is no longer declared in SCHEMA; re-check this invariant"
        assert "made_before_tip" not in declared.group(1), (
            f"{table} grew a stored timing flag; it must be derived, never stored"
        )


def test_a_rerun_before_tip_is_recorded_but_the_display_still_serves_the_freshest_read(tmp_path):
    """The two rules are different on purpose, and the difference is pinned.

    The RECORD counts the earliest recorded pick -- what the model said first,
    which is what the record means. The DISPLAY (`picks_by_player_stat`, and so
    the +/- beside a projection) serves the newest pre-tip read, because a
    number printed next to a box score has to be the freshest thing the model
    said about it.

    Both answers are the same when a game has one pre-tip pick, which is almost
    always. This test is the one place they are not, and it states the
    difference rather than leaving a reader to infer it: an earlier PR would
    have had to pick one rule and the other surface would have been quietly
    wrong.
    """
    db = _db(tmp_path)
    store.insert_player_prediction(
        db, game_id="g1", player_id="p1", stat="points",
        predicted_value=20.0, created_at="2026-03-01T09:00:00+00:00",
    )
    store.insert_player_prediction(
        db, game_id="g1", player_id="p1", stat="points",
        predicted_value=31.0, created_at="2026-03-01T10:00:00+00:00",
    )
    store.insert_player_outcome(
        db, game_id="g1", player_id="p1", stat="points",
        actual_value=24.0, recorded_at="2026-03-02T09:00:00+00:00",
    )
    game = _completed("g1", "BOS", "MIA", 110, 100)

    from nba_predictor.tracking.player_props import picks_by_player_stat
    served, rebuilt = picks_by_player_stat(db, "g1", game)[("p1", "points")]

    assert served["predicted_value"] == 31.0, "the ranking must show the freshest pre-tip read"
    assert rebuilt is False
    # The record graded the FIRST one, which is the assertion that the two
    # rules are not the same rule wearing one name.
    graded = resolved_player_props(db, [game])
    assert [r["predicted_value"] for r in graded] == [20.0]


# --- the shipped state: is the headline empty over a populated record? --------

def test_the_shipped_NBA_record_is_empty_and_has_nothing_hidden_underneath():
    """PL's shipped payload was `n_resolved_fixtures: 0, pct_correct_overall:
    null` with 50 graded picks sitting in the same file. NBA's shape is checked
    here so the difference is on the record rather than assumed: the shipped
    snapshot's `track_record` and the shipped database are BOTH empty, so
    there is no populated record hiding under an empty headline.

    If a future snapshot ships rows, this fails and whoever lands it has to say
    whether the headline is populated and what its n is -- which is the whole
    point of the check.
    """
    snapshot = json.loads((REPO / "data" / "public_snapshot.json").read_text())
    shipped_track_record = snapshot.get("track_record")

    db = REPO / "data" / "tracking.db"
    if not db.exists():
        pytest.skip("no shipped tracking database in this checkout")

    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        stored = {
            table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("predictions", "game_market_predictions",
                          "player_prediction_snapshots", "game_player_outcomes")
        }

    assert stored == {t: 0 for t in stored}, f"the shipped database now holds picks: {stored}"
    assert shipped_track_record == [], (
        f"the shipped snapshot now carries a track record: {shipped_track_record}. "
        "Report the headline's n and hit rate here before changing this assertion."
    )
