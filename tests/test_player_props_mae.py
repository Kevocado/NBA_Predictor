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
