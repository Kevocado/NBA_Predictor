"""Rest days measured from the team's last game WHATEVER its role (opt-in), versus the role-split default.

The default measures a home game's rest from the team's last HOME game and an away game's from its last AWAY game, so
DEN at home on the 3rd and away on the 4th looks rested on the 4th (and a team's first away game of the season reads
99 days). Every committed model was fitted on that default, so the correction is opt-in until it has been evaluated.
"""
import pandas as pd
import pytest

from nba_predictor.features.build import build_feature_frame
from tests.test_travel_fatigue_features import _trip_frame


def _row(frame, game_id):
    return frame[frame["game_id"] == game_id].iloc[0]


def test_default_is_unchanged_role_split():
    default, _ = build_feature_frame(_trip_frame())
    explicit, _ = build_feature_frame(_trip_frame(), rest_by_team=False)
    pd.testing.assert_frame_equal(default, explicit)


def test_default_misses_the_back_to_back_that_opt_in_sees():
    default, _ = build_feature_frame(_trip_frame())
    fixed, _ = build_feature_frame(_trip_frame(), rest_by_team=True)
    # m1: DEN hosts LAL on 2026-01-03; m2: LAL hosts DEN on 2026-01-04, so DEN plays two nights in a row.
    assert _row(fixed, "m2")["away_rest_days"] == 0
    assert bool(_row(fixed, "m2")["away_back_to_back"]) is True
    assert _row(default, "m2")["away_rest_days"] > 0           # the defect the opt-in fixes
    assert bool(_row(default, "m2")["away_back_to_back"]) is False


def test_rest_by_team_can_only_shorten_rest():
    """The team's last game of ANY role is at least as recent as its last game in one role."""
    default, _ = build_feature_frame(_trip_frame())
    fixed, _ = build_feature_frame(_trip_frame(), rest_by_team=True)
    both = default.merge(fixed, on="game_id", suffixes=("_d", "_f"))
    assert len(both) == len(default) == len(fixed)
    assert (both["home_rest_days_f"] <= both["home_rest_days_d"]).all()
    assert (both["away_rest_days_f"] <= both["away_rest_days_d"]).all()
    assert (both["home_rest_days_f"] < both["home_rest_days_d"]).any() or (both["away_rest_days_f"] < both["away_rest_days_d"]).any()
