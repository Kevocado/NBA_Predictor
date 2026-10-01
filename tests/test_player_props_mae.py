"""In-sample MAE is a grouped aggregate over RESOLVED rows only.

The measured reality on main: models/player_props.py is an XGBRegressor
returning a raw predicted_value, and PlayerPropOut carried no error estimate
at all, so a picks row had nothing honest to put after the "+/-". This is the
"computed from data that exists" half.

Two rules the tests pin, both because the wrong alternative is a lie:

  * a prediction with no actual_value is not resolved and never contributes
    (counting it would drag the error toward the model's own guesses);
  * a stat with no resolved rows is None, never 0.0 -- 0.0 would claim the
    model never missed by a tenth of a point, which is a different statement
    from "never measured". schemas.py already reasons this way for
    TrackRecordOut.hit_rate.
"""

import pytest

from nba_predictor.models import player_props


def _rows(*triples):
    return [
        {"stat": stat, "predicted_value": predicted, "actual_value": actual}
        for stat, predicted, actual in triples
    ]


def test_mae_is_the_mean_absolute_error_of_the_resolved_rows():
    rows = _rows(
        ("points", 20.0, 24.0),  # off by 4
        ("points", 10.0, 14.0),  # off by 4
        ("points", 30.0, 27.0),  # off by 3
    )

    assert player_props.in_sample_mae_by_stat(rows)["points"] == pytest.approx(11 / 3)


def test_mae_is_grouped_per_stat_not_pooled_across_stats():
    rows = _rows(
        ("points", 20.0, 24.0),  # points: 4
        ("threes", 3.0, 3.0),  # threes: 0
    )

    result = player_props.in_sample_mae_by_stat(rows)

    assert result["points"] == pytest.approx(4.0)
    assert result["threes"] == pytest.approx(0.0)
    # A real zero (threes was exactly right) is distinct from "no measurement".
    assert result["threes"] is not None


def test_a_prediction_with_no_actual_does_not_contribute_to_the_aggregate():
    rows = _rows(
        ("points", 20.0, 24.0),  # off by 4
        ("points", 99.0, None),  # unresolved: must not drag the mean
    )

    assert player_props.in_sample_mae_by_stat(rows)["points"] == pytest.approx(4.0)


def test_a_stat_with_no_resolved_rows_is_null_not_zero():
    rows = _rows(("points", 20.0, 24.0), ("threes", 3.0, None))

    result = player_props.in_sample_mae_by_stat(rows)

    assert result["threes"] is None
    assert result["threes"] != 0.0


def test_no_resolved_rows_at_all_is_null_for_every_stat():
    result = player_props.in_sample_mae_by_stat([])

    assert result == {"points": None, "rebounds": None, "assists": None, "threes": None}


def test_no_rows_at_all_is_null_for_every_stat():
    result = player_props.in_sample_mae_by_stat(None)

    assert result == {"points": None, "rebounds": None, "assists": None, "threes": None}


def test_mae_covers_exactly_the_stats_the_models_target():
    # The keys are the model's own STAT_TARGETS, so a row can never be served
    # without an error estimate for its own stat.
    assert set(player_props.in_sample_mae_by_stat([])) == set(player_props.STAT_TARGETS)


def test_mae_is_a_mean_of_magnitudes_so_signs_cannot_cancel():
    rows = _rows(("rebounds", 10.0, 4.0), ("rebounds", 10.0, 16.0))  # -6 and +6

    # A signed mean would be 0.0 and would claim the model was perfect.
    assert player_props.in_sample_mae_by_stat(rows)["rebounds"] == pytest.approx(6.0)


def test_actual_only_rows_do_not_create_a_stat_key():
    # A resolved row whose prediction is missing cannot be scored; it must not
    # open a new stat bucket that then reads as measured.
    rows = [{"stat": "assists", "predicted_value": None, "actual_value": 7.0}]

    result = player_props.in_sample_mae_by_stat(rows)

    assert result["assists"] is None


def test_actual_only_row_does_not_disturb_the_other_resolved_rows():
    rows = _rows(("assists", 6.0, 9.0)) + [
        {"stat": "assists", "predicted_value": None, "actual_value": 30.0}
    ]

    assert player_props.in_sample_mae_by_stat(rows)["assists"] == pytest.approx(3.0)


def test_the_mae_is_not_rebuilt_from_the_whole_schedule_on_every_request(tmp_path, monkeypatch):
    """/games/{id}/players called resolved_player_props over the ENTIRE schedule per
    request: one outcomes query and one predictions query per scheduled game (about
    3,500 queries) with nothing cached (CodeRabbit on NBA#18). The result only changes
    when the tracking database does, so it is computed once per database state."""
    import os
    from nba_predictor.api import routes

    db = tmp_path / "tracking.db"
    db.write_text("x")
    calls = []

    def _resolved(db_path, schedule):
        calls.append(1)
        return []

    monkeypatch.setattr(routes, "resolved_player_props", _resolved)
    routes._MAE_CACHE.clear()
    schedule = [{"game_id": "1"}, {"game_id": "2"}]
    routes._mae_by_stat(db, schedule)
    routes._mae_by_stat(db, schedule)
    routes._mae_by_stat(db, schedule)
    assert len(calls) == 1, f"the MAE was recomputed {len(calls)} times for an unchanged database"

    # The database changed (a game resolved): the next request recomputes, once.
    db.write_text("xy")
    os.utime(db, None)
    routes._mae_by_stat(db, schedule)
    routes._mae_by_stat(db, schedule)
    assert len(calls) == 2, "a changed tracking database must invalidate the cached MAE"

    # A different schedule is a different question.
    routes._mae_by_stat(db, schedule + [{"game_id": "3"}])
    assert len(calls) == 3


def _reference_resolved(db_path, schedule):
    """The per-game algorithm, kept here as the specification of the bulk one.

    Rewritten with the bulk version on 2026-10-01 (predictor-hub #66). The old
    reference went through `picks_by_player_stat` and dropped every `rebuilt`
    row, so it encoded the exclusion this change removed: for its `_seed` below
    it returned ONE row (points 20.0 -> 24.0), because game B's only pick was a
    backtest row. The new reference counts one row per (game, player, stat) --
    the EARLIEST recorded, whenever it was made -- and returns TWO.

    `picks_by_player_stat` is still the DISPLAY rule (newest pre-tip row, else
    the newest) and is no longer what the MAE grades. That is deliberate and
    documented at both definitions.
    """
    from nba_predictor.tracking.player_props import actuals_by_player_stat
    from nba_predictor.tracking.timing import earliest_recorded
    from nba_predictor.tracking import store

    out = []
    for game in schedule:
        gid = game.get("game_id")
        if not gid:
            continue
        actuals = actuals_by_player_stat(db_path, gid)
        rows = store.get_player_predictions_for_game(db_path, gid)
        for pick in earliest_recorded(rows):
            actual = actuals.get((pick["player_id"], pick["stat"]))
            if actual is None:
                continue
            out.append({"game_id": gid, "player_id": pick["player_id"], "stat": pick["stat"],
                        "predicted_value": pick["predicted_value"], "actual_value": actual,
                        "created_at": pick["created_at"]})
    return out


def _seed(tmp_path):
    from nba_predictor.tracking import store
    db = tmp_path / "tracking.db"
    store.init_db(db)
    # Game A: a pre-tip pick (counts), a later rerun of it (history, not graded),
    # and an actual. The rerun is the better guess (99.0 vs 20.0) and must not
    # be the graded one, or re-running the model until it looked right is free.
    store.insert_player_prediction(db, game_id="A", player_id="p1", stat="points", predicted_value=20.0, created_at="2026-10-01T10:00:00Z")
    store.insert_player_prediction(db, game_id="A", player_id="p1", stat="points", predicted_value=99.0, created_at="2026-10-02T10:00:00Z")
    store.insert_player_outcome(db, game_id="A", player_id="p1", stat="points", actual_value=24.0, recorded_at="2026-10-02T09:00:00Z")
    # Game B: only a post-tip (rebuilt) row, with an actual. Excluded before
    # 2026-10-01; a recorded pick now.
    store.insert_player_prediction(db, game_id="B", player_id="p2", stat="assists", predicted_value=7.0, created_at="2026-10-03T10:00:00Z")
    store.insert_player_outcome(db, game_id="B", player_id="p2", stat="assists", actual_value=3.0, recorded_at="2026-10-03T11:00:00Z")
    # Game C: pick before tip, NO actual yet -> unresolved.
    store.insert_player_prediction(db, game_id="C", player_id="p3", stat="rebounds", predicted_value=9.0, created_at="2026-10-01T10:00:00Z")
    # Game D: pick before tip, actual present.
    store.insert_player_prediction(db, game_id="D", player_id="p4", stat="points", predicted_value=15.0, created_at="2026-10-01T10:00:00Z")
    store.insert_player_outcome(db, game_id="D", player_id="p4", stat="points", actual_value=18.0, recorded_at="2026-10-02T09:00:00Z")
    games = {g: {"game_id": g, "game_date": "2026-10-02", "tip_off": "2026-10-02T00:00:00Z"} for g in "ABCD"}
    # A season's worth of unplayed games with no rows at all.
    schedule = list(games.values()) + [{"game_id": f"F{i}", "game_date": "2027-03-01"} for i in range(1756)]
    return db, schedule


def test_the_bulk_resolution_returns_exactly_what_the_per_game_algorithm_did(tmp_path):
    from nba_predictor.tracking.player_props import resolved_player_props
    db, schedule = _seed(tmp_path)
    expected = sorted(_reference_resolved(db, schedule), key=lambda r: (r["stat"], r["predicted_value"]))
    got = sorted(resolved_player_props(db, schedule), key=lambda r: (r["stat"], r["predicted_value"]))
    assert [{k: v for k, v in r.items() if k != "made_before_tip"} for r in got] == expected
    # Three rows, one per (game, player, stat): game A's EARLIEST points pick
    # (20.0, not the 99.0 rerun), game B's post-tip assists, game D's points.
    # The old rule returned two of these -- points 20.0 and points 15.0 -- and
    # dropped game B entirely.
    assert [(r["game_id"], r["stat"], r["predicted_value"], r["actual_value"]) for r in got] == [
        ("B", "assists", 7.0, 3.0),      # sorted by (stat, predicted_value)
        ("D", "points", 15.0, 18.0),
        ("A", "points", 20.0, 24.0),     # A's EARLIEST pick, not the 99.0 rerun
    ]
    # And the disclosure is on the row, so the two figures need no second pass.
    assert {(r["game_id"], r["made_before_tip"]) for r in got} == {("A", True), ("B", False), ("D", True)}


def test_resolving_a_full_season_does_not_open_a_connection_per_scheduled_game(tmp_path, monkeypatch):
    """~1,760 scheduled games used to mean ~3,500 queries, ~50 s on the live site for
    the first visitor after any database change."""
    from nba_predictor.tracking import player_props as tp, store
    db, schedule = _seed(tmp_path)
    opened = []
    real = store.get_connection

    def _counting(path):
        opened.append(1)
        return real(path)

    monkeypatch.setattr(store, "get_connection", _counting)
    tp.resolved_player_props(db, schedule)
    assert len(opened) <= 4, f"{len(opened)} connections opened for a {len(schedule)}-game schedule"
