"""The prop models' feature frame: his own rolling means + minutes + matchup + home/away, causal, NaN-tolerant."""
import numpy as np
import pandas as pd

from nba_predictor.features.prop_model_frame import PROP_FEATURE_COLUMNS, build_prop_model_frame


def _world(n=16, upcoming_position="SG"):
    rows = []
    for i in range(n):
        day = f"2026-03-{1 + i:02d}"
        home = i % 2 == 0
        for pid, pos, pts, minutes in (("a", "G", 20.0 + i, 30.0 + i % 3), ("b", "G", 8.0, 20.0), ("c", "C", 12.0, 25.0)):
            for team, opp in (("X", "Y"), ("Y", "X")):
                rows.append(dict(player_id=f"{pid}{team}", player_name=pid, team=team, opponent=opp, game_id=f"g{i}", game_date=day,
                                 points=pts, rebounds=5.0, assists=3.0, fg3m=1.0, minutes=minutes, position=pos, fga=10.0,
                                 home_team="X" if home else "Y", away_team="Y" if home else "X"))
    up = dict(player_id="aX", player_name="a", team="X", opponent="Y", game_id="next", game_date="2026-03-30", points=None, rebounds=None,
              assists=None, fg3m=None, minutes=None, position=upcoming_position, fga=None, home_team="X", away_team="Y")
    return pd.DataFrame(rows + [up])


def test_the_frame_carries_minutes_matchup_and_home_columns():
    frame, cols = build_prop_model_frame(_world())
    assert cols == PROP_FEATURE_COLUMNS
    assert {"min_last", "min_ewm5", "minutes_trend", "opp_def_vs_pos", "rest_days", "usage_trend", "is_home"} <= set(cols)
    row = frame[(frame.player_id == "aX") & frame.points.isna()].iloc[0]
    assert row["is_home"] == 1.0
    assert row["min_last"] == 30.0 + 15 % 3                  # his previous game's minutes, nothing from the game being predicted
    assert row["opp_def_vs_pos"] == row["opp_def_vs_pos"]    # not NaN: enough history for Y against guards


def test_a_roster_position_abbreviation_counts_as_its_box_score_group():
    sg, _ = build_prop_model_frame(_world(upcoming_position="SG"))
    g, _ = build_prop_model_frame(_world(upcoming_position="G"))
    up = lambda f: f[(f.player_id == "aX") & f.points.isna()].iloc[0]["opp_def_vs_pos"]
    assert up(sg) == up(g) and not np.isnan(up(sg))


def test_the_upcoming_rows_features_do_not_depend_on_its_own_result():
    base = _world()
    leaked = base.copy()
    leaked.loc[leaked.game_id == "next", ["points", "minutes"]] = [99.0, 48.0]     # pretend the game was played
    f1, cols = build_prop_model_frame(base)
    f2, _ = build_prop_model_frame(leaked)
    r1 = f1[(f1.player_id == "aX") & (f1.game_id == "next")].iloc[0][cols]
    r2 = f2[(f2.player_id == "aX") & (f2.game_id == "next")].iloc[0][cols]
    pd.testing.assert_series_equal(r1.astype(float), r2.astype(float))
