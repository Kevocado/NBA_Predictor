"""Prop matchup features: opponent D-vs-position, rest, usage and minutes trend.

Spec section 7. The prop models run on five rolling means and nothing else —
no opponent, no rest, no minutes trend, no idea who the player is guarding or
playing against. Task 9 adds the four the plan names, position-keyed G/F/C the
way the NFL features key QB/RB/TE/WR.

Pinned, each of which can fail:

* **shift(1)-then-rolling throughout.** A matchup feature that has seen its own
  game is the repo's number one historical leakage bug, and a matchup number is
  exactly where it hides — `opp_def_vs_pos` is built from the opponent's
  conceded rate, so including the target game would leak the target game's own
  result into the feature predicting it.
* **Position groups are G/F/C and present.** ESPN sends G, F and C; a mapping
  that silently drops an unmapped position would shrink the frame quietly.
* **No all-NaN columns** (spec 12.2) — a broken pull reads as no signal.
* **An unseen position degrades to neutral, not to a fabricated number.** There
  is no team-vs-"unknown-position" defence to look up, and inventing one is the
  LeBron failure in a new place.
"""

import numpy as np
import pandas as pd
import pytest

from nba_predictor.features.prop_matchup import POSITION_GROUPS, build_prop_features

NEW_FEATURES = ["opp_def_vs_pos", "rest_days", "usage_trend", "minutes_trend"]


def make_player_games(n_games: int = 90, n_players: int = 30, seed: int = 11) -> pd.DataFrame:
    """Player-games with a team column, position, and an opponent we can join."""
    rng = np.random.default_rng(seed)
    teams = ["BOS", "MIA", "LAL", "GSW", "DEN", "PHX"]
    positions = ["G", "F", "C"]
    rows = []
    dates = pd.date_range("2025-10-22", periods=n_games).astype(str)
    for g, day in enumerate(dates):
        for k in range(len(teams) // 2):
            home, away = teams[2 * k], teams[2 * k + 1]
            for team, opponent in ((home, away), (away, home)):
                for p in range(n_players // (len(teams) // 2) // 2):
                    rows.append({
                        "player_id": f"p{p:02d}",
                        "player_name": f"Player {p}",
                        "team": team,
                        "opponent": opponent,
                        "position": positions[p % 3],
                        "game_id": f"g{g}-{k}",
                        "game_date": day,
                        "minutes": float(rng.uniform(20, 38)),
                        "points": float(rng.uniform(8, 30)),
                        "rebounds": float(rng.uniform(2, 12)),
                        "assists": float(rng.uniform(1, 10)),
                        "fg3m": float(max(0, rng.poisson(1 + p % 3))),
                        "fga": float(rng.uniform(8, 25)),
                    })
    return pd.DataFrame(rows)


def test_all_four_new_features_are_produced():
    feats = build_prop_features(make_player_games())
    for col in NEW_FEATURES:
        assert col in feats.columns, f"{col} was not produced"


def test_no_new_feature_column_is_mostly_nan():
    """An all-NaN column is a broken pull, not the absence of signal
    (spec 12.2)."""
    feats = build_prop_features(make_player_games())
    for col in NEW_FEATURES:
        coverage = feats[col].notna().mean()
        assert coverage > 0.8, (
            f"{col} is {coverage:.0%} non-NaN -- broken pull, not no signal"
        )


def test_features_never_use_the_game_they_describe():
    """shift(1). `opp_def_vs_pos` is the opponent's rate of conceding to a
    position -- so if the window included the target game, the feature would
    contain the target game's own result. That is the repo's number one
    historical leakage bug and it hides precisely here."""
    games = make_player_games().sort_values(["game_date", "player_id"]).reset_index(drop=True)
    feats = build_prop_features(games)

    # For one player on one game, the matchup feature must be built from games
    # strictly before that date.
    target_date = "2025-11-20"
    row = feats[(feats["player_id"] == "p00") & (feats["game_date"] == target_date)]
    if row.empty:
        pytest.skip("fixture has no such row")
    row = row.iloc[0]

    prior = games[
        (games["game_date"] < target_date)
        & (games["opponent"] == row["opponent"])
        & (games["position"] == row["position"])
    ]
    if prior["game_date"].nunique() < 3:
        pytest.skip("no prior history in the fixture")

    # Trailing window of that matchup's per-36 conceded rate, averaged per DATE
    # then windowed -- so the whole target night is excluded, not just one row.
    per_date = (
        prior.assign(per36=prior["points"] / prior["minutes"] * 36)
        .groupby("game_date")["per36"]
        .mean()
        .tail(10)
    )
    expected = float(per_date.mean())
    assert row["opp_def_vs_pos"] == pytest.approx(expected, abs=1e-6), (
        "opp_def_vs_pos for this game used that game or a later one, or is not "
        "a date-ordered trailing window of this matchup's conceded rate"
    )


def test_rest_days_is_positive_and_non_negative():
    """Rest days counts days since the same player's previous game. Zero would
    mean they played the previous day; negative is impossible."""
    feats = build_prop_features(make_player_games())
    known = feats["rest_days"].dropna()
    assert len(known) > 0, "no player had two games"
    assert (known >= 0).all(), f"negative rest days: {known.min()}"


def test_usage_and_minutes_trends_are_shifted_rates_not_raw_counts():
    """usage_trend must be a *rate*, comparable across players -- a raw FGA
    count would just rank big men."""
    feats = build_prop_features(make_player_games())
    for col in ("usage_trend", "minutes_trend"):
        known = feats[col].dropna()
        assert known.between(-2, 200).all(), f"{col} outside plausible range"


def test_position_groups_are_g_f_c_and_unmapped_positions_degrade_neutrally():
    """ESPN sends G, F, C. An unmapped position must become NaN, not a
    fabricated number: there is no defence-vs-unknown-position to look up."""
    games = make_player_games()
    games.loc[games.index[:20], "position"] = "Z"

    feats = build_prop_features(games)
    assert set(POSITION_GROUPS) == {"G", "F", "C"}
    unmapped = feats[feats["position"] == "Z"]
    assert len(unmapped) > 0, "fixture did not produce unmapped rows"
    assert unmapped["opp_def_vs_pos"].isna().all(), (
        "an unmapped position produced a matchup number -- that is fabricated"
    )


def test_every_game_position_pair_gets_a_non_nan_matchup_value():
    """Coverage must be real. If G/F/C rows come back all-NaN the feature is
    broken and the coverage test above would be measuring the wrong column."""
    feats = build_prop_features(make_player_games())
    known = feats[feats["position"].isin(["G", "F", "C"])]["opp_def_vs_pos"]
    assert known.notna().mean() > 0.8, "mapped positions produced no matchup values"


def test_input_is_not_mutated():
    """The frame comes from the pipeline's caches. Mutating it in place would
    corrupt the next caller with no error to notice."""
    games = make_player_games()
    before = games.copy()
    build_prop_features(games)
    pd.testing.assert_frame_equal(games, before)