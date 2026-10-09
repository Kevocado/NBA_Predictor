"""Task 4(b): rest, back-to-back and travel interactions, actually computed.

`home_fatigue_index` and `away_fatigue_index` have been in `FEATURE_COLUMNS` since
the column list was written, and `_OPTIONAL_COLUMNS_DEFAULT_ZERO` fills them with
0.0 when they are absent. Nothing computed them. So a feature the model was
already told to expect has been a constant zero, which is worse than not having
the column: it reads as a measurement and means nothing.

`rest_travel.py` already has every function needed -- `rolling_travel_miles`,
`timezone_change_count`, `congestion_flags`, `fatigue_index`. They were written
and then never called. This wires them in.

**Causality is the whole point of the feature**, so it is what the tests check
first: a team's travel is built from ITS OWN games strictly before the one being
scored, never from the game itself and never from the opponent's later games.
"""
from __future__ import annotations

import pandas as pd
import pytest

from nba_predictor.features.build import build_feature_frame

BOX = {"fgm": 40, "fga": 88, "fg3m": 12, "tov": 11, "oreb": 9, "dreb": 32, "fta": 20}


def _game(game_id, date, home, away, **over):
    row = {
        "game_id": game_id, "game_date": date,
        "home_team": home, "away_team": away,
        "home_pts": 110, "away_pts": 100, "home_win": 1,
    }
    for field, value in BOX.items():
        row[f"home_{field}"] = value
        row[f"away_{field}"] = value
    row.update(over)
    return row


# DEN->LAL is a real intra-conference trip; LAL->ATL is a long one. The exact
# miles are not the point -- the ordering and the non-zero-ness are.
def _trip_frame():
    """Warm-up games so every team's rolling features exist AND every team has
    travelled, then the measured games.

    Two things a naive fixture gets wrong, both of which make the test pass for
    the wrong reason:

    * a team's FIRST game has NaN rolling features, and `build_feature_frame`
      drops rows with any NaN feature column — so a short fixture drops nearly
      everything and the assertions end up on an empty frame;
    * if the warm-up is all home games, travel is zero by construction and
      "the feature moved" would be untestable.

    So the warm-up is a round robin where each team hosts and travels.
    """
    rows = []
    teams = ["DEN", "LAL", "ATL", "BOS"]
    for r in range(4):
        # swap the host each round so every team plays both home and away
        pairs = [(0, 1), (2, 3)] if r % 2 == 0 else [(1, 0), (3, 2)]
        for i, (h, a) in enumerate(pairs):
            rows.append(_game(
                f"w{r}-{i}", f"2025-12-{r * 3 + i * 2 + 1:02d}", teams[h], teams[a]))
    measured = [
        ("m1", "2026-01-03", "DEN", "LAL"),   # DEN's first measured game
        ("m2", "2026-01-04", "LAL", "DEN"),   # both travelling
        ("m3", "2026-01-06", "DEN", "ATL"),   # DEN home on day 3 of 3: three in four
        ("m4", "2026-01-20", "DEN", "BOS"),   # 14 days of rest, window clear
    ]
    rows.extend(_game(*m) for m in measured)
    return pd.DataFrame(rows)


MEASURED = ["m1", "m2", "m3", "m4"]


def test_fatigue_index_is_not_a_constant_zero():
    """The defect this file exists to fix.

    `_OPTIONAL_COLUMNS_DEFAULT_ZERO` fills these with 0.0 when nothing computes
    them, so a constant-zero column passes every check that only looks at
    whether the column exists.
    """
    frame, _ = build_feature_frame(_trip_frame())
    assert "home_fatigue_index" in frame.columns

    values = frame["home_fatigue_index"].to_numpy()
    assert len(values) == len(frame)
    assert not (values == 0.0).all(), (
        f"every home_fatigue_index is 0.0 -- the column is still a constant, which is "
        "what _OPTIONAL_COLUMNS_DEFAULT_ZERO does when nothing computes it"
    )


def test_the_same_holds_for_the_away_side():
    frame, _ = build_feature_frame(_trip_frame())
    away = frame["away_fatigue_index"].to_numpy()
    assert not (away == 0.0).all(), "every away_fatigue_index is 0.0"


def test_the_trip_into_the_game_being_scored_counts():
    """CodeRabbit Major: the path used to end at the team's PREVIOUS venue, so DEN at home on the 3rd and at LAL
    on the 4th recorded no DEN->LAL leg. The path ends at THIS game's venue (the home team's arena, known
    before tip-off)."""
    from nba_predictor.features.rest_travel import rolling_travel_miles

    frame, _ = build_feature_frame(_trip_frame())
    m2 = frame[frame["game_id"] == "m2"].iloc[0]          # LAL hosts DEN the day after DEN hosted LAL
    assert m2["away_travel_miles"] >= rolling_travel_miles(["DEN", "LAL"]) > 0


def test_fatigue_rest_is_measured_from_the_teams_last_game_home_or_away():
    """CodeRabbit Major: home and away last-game dates were tracked separately, so DEN at home on the 3rd and away
    on the 4th looked rested on the 4th. The fatigue candidate must see the back-to-back."""
    from nba_predictor.features.rest_travel import (
        compute_rest_days, fatigue_index, rolling_travel_miles, timezone_change_count)

    frame, _ = build_feature_frame(_trip_frame())
    m2 = frame[frame["game_id"] == "m2"].iloc[0]
    # DEN's path in the 7 days to m2 ends DEN (m1 venue) -> LAL (m2 venue), after earlier venues; recompute it
    # from the frame's own columns instead of re-deriving the whole window.
    rest = compute_rest_days("2026-01-04", "2026-01-03")
    expected = fatigue_index(m2["away_travel_miles"], m2["away_timezone_changes"], rest)
    assert m2["away_fatigue_index"] == pytest.approx(expected)
    assert rest <= 1


def test_the_candidate_columns_are_not_in_the_default_model_contract():
    """CodeRabbit Major: the candidate loses its evaluation, so a default retrain must not pick it up."""
    from nba_predictor.features.build import FEATURE_COLUMNS, TRAVEL_FATIGUE_COLUMNS, build_training_frame

    assert not set(TRAVEL_FATIGUE_COLUMNS) & set(FEATURE_COLUMNS)
    _, default_cols = build_training_frame(_trip_frame())
    assert not set(TRAVEL_FATIGUE_COLUMNS) & set(default_cols)
    _, opted_in = build_training_frame(_trip_frame(), include_travel_fatigue=True)
    assert set(TRAVEL_FATIGUE_COLUMNS) <= set(opted_in)


def test_travel_uses_only_the_teams_own_prior_games():
    """Causality. A team's travel is built from ITS OWN games strictly before the
    one being scored.

    DEN's game 3 is at home. Its travel must come from games 1 and 2 only. Add a
    game AFTER it and game 3's features must not move.
    """
    frame, _ = build_feature_frame(_trip_frame())
    before = frame.set_index("game_id").loc[["m1", "m2", "m3", "m4"]][
        ["home_travel_miles", "home_fatigue_index"]]

    later = _trip_frame()
    later = pd.concat([later, pd.DataFrame([_game("m5", "2026-01-22", "DEN", "LAL")])],
                      ignore_index=True)
    after = build_feature_frame(later)[0].set_index("game_id").loc[
        ["m1", "m2", "m3", "m4"]][["home_travel_miles", "home_fatigue_index"]]

    pd.testing.assert_frame_equal(before, after, check_names=False)
    assert not (before["home_travel_miles"] == 0.0).all(), (
        "the fixture produces no travel at all, so the causality check is vacuous"
    )


def test_a_long_rest_clears_the_travel_window():
    """The window is 7 days, not 'all history': a team rested for two weeks is not
    carrying last month's road trip."""
    frame, _ = build_feature_frame(_trip_frame())
    rested = frame[frame.game_id == "m4"].iloc[0]
    assert rested["home_travel_miles"] == 0.0, (
        f"a game after 14 days of rest still reports {rested['home_travel_miles']} "
        "miles of travel: the 7-day window is not being applied"
    )


def test_congestion_counts_games_in_the_window():
    """three-in-four: DEN plays on the 3rd, 4th and 6th, so by the 6th it has
    played three games inside four days.

    The window is four days ENDING on the game being scored, so a game on the 1st
    is outside it — the earlier version of this test asserted the wrong dates.
    """
    frame, _ = build_feature_frame(_trip_frame())
    congested = frame[frame.game_id == "m3"].iloc[0]
    assert congested["home_three_in_four"] is True or congested["home_three_in_four"] == 1, (
        "DEN plays on the 3rd, 4th and 6th and is not flagged three-in-four"
    )
    rested = frame[frame.game_id == "m4"].iloc[0]
    assert not rested["home_three_in_four"], "a game after 14 days of rest is still congested"


def test_the_fatigue_index_rises_with_load():
    """The feature has to move in the right direction, or it is noise with a name."""
    frame, _ = build_feature_frame(_trip_frame())
    by_id = frame.set_index("game_id")
    loaded = by_id.loc["m3", "home_fatigue_index"]
    rested = by_id.loc["m4", "home_fatigue_index"]
    assert loaded > rested, (
        f"the loaded game ({loaded}) does not carry more fatigue than the rested "
        f"one ({rested}); the index is not moving with load"
    )


def test_both_sides_are_computed_independently():
    """Home and away fatigue are different teams' loads, not one number shared."""
    frame, _ = build_feature_frame(_trip_frame())
    both = frame[(frame.home_fatigue_index != 0) | (frame.away_fatigue_index != 0)]
    assert len(both), "neither side ever has fatigue"
    # And where the two teams' situations differ, the numbers must differ.
    differing = frame[frame.home_fatigue_index != frame.away_fatigue_index]
    assert len(differing), (
        "home and away fatigue are always equal: the two sides are being given the "
        "same number rather than their own"
    )
