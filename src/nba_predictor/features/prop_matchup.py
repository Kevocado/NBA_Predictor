"""Prop matchup features: opponent D-vs-position, rest, usage and minutes trend.

Spec section 7. The prop models run on five rolling means of the player's own
stats and nothing else — no opponent, no rest, no minutes trend. This adds the
four the plan names, position-keyed G/F/C the way the NFL features key
QB/RB/TE/WR.

Everything is shift(1)-then-rolling, which matters most for
`opp_def_vs_pos`. That feature is the rate at which the *opponent* concedes to a
position, so a window that included the target game would put the target game's
own result inside the feature predicting it. The date guard is in the tests, not
just the slicing.

`opp_def_vs_pos` is expressed per 36 minutes, because a raw per-game rate is not
comparable between a player who logged 38 minutes and one who logged 14.

**An unmapped position yields NaN, not a number.** ESPN sends G, F and C; there
is no defence-vs-unknown-position to look up, and inventing one is the LeBron
fabrication in a new place (spec 12.4). Same for a player's first game: no
history means NaN, because 0.0 would claim a measured average of zero.

The input frame is copied rather than mutated — it comes from the pipeline's
caches, and an in-place edit corrupts the next caller with nothing to notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: ESPN's abbreviations, mapped to the position groups the plan names. Keys are
#: what ESPN sends; values are the group. A position outside this map is not
#: guessed at -- it produces NaN features.
POSITION_GROUPS = {"G": "G", "F": "F", "C": "C"}

#: Trailing window for the rate and trend features, matching the repo's existing
#: 10-game rolling convention in features/player_stats.py.
WINDOW = 10

#: Per-36-minute basis, so a rate is comparable across minutes played.
PER_36 = 36.0

NEW_FEATURE_COLUMNS = ["opp_def_vs_pos", "rest_days", "usage_trend", "minutes_trend"]


def _position_group(position) -> str | None:
    if not isinstance(position, str):
        return None
    return POSITION_GROUPS.get(position.strip().upper())


def build_prop_features(
    player_games: pd.DataFrame,
    window: int = WINDOW,
) -> pd.DataFrame:
    """Add matchup, rest and trend columns to a player-game frame.

    Returns a copy; the input is left untouched.
    """
    required = {"player_id", "game_date", "team", "opponent", "position"}
    missing = required - set(player_games.columns)
    if missing:
        raise ValueError(
            f"player-game frame is missing {sorted(missing)}; "
            "opp_def_vs_pos needs a team and an opponent to be a matchup"
        )

    games = player_games.copy()
    games["position_group"] = games["position"].map(_position_group)
    games = games.sort_values(["player_id", "game_date"]).reset_index(drop=True)

    # ---- rest days: since this player's previous game, for this player only.
    dates = pd.to_datetime(games["game_date"])
    games["rest_days"] = (
        dates - dates.groupby(games["player_id"]).shift(1)
    ).dt.days

    # ---- usage_trend: share of the team's attempts, trailing.
    # A rate, not a count: raw FGA would only rank big men.
    #
    # NaN, not zero, when the frame carries no `fga`. A usage share computed
    # without shot volume is not a small number, it is a wrong one, and an
    # all-NaN column here is the honest signal that the extract is missing
    # something (spec 12.2) rather than that players have no usage.
    if "fga" in games.columns:
        team_fga = games.groupby(["game_id", "team"])["fga"].transform("sum")
        games["_usage"] = games["fga"] / team_fga.replace(0, np.nan)
    else:
        games["_usage"] = np.nan

    games["_usage_trend"] = games.groupby("player_id")["_usage"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=2).mean()
    )
    games["usage_trend"] = games["_usage_trend"]

    # Written to its own column, never back over `minutes`. An earlier draft
    # looped over ("_usage", "minutes") and assigned the trailing mean into
    # `minutes` itself, which silently replaced a raw input column with a
    # rolling average -- and then fed that corrupted value into opp_def_vs_pos
    # through the per-36 calculation below.
    games["minutes_trend"] = games.groupby("player_id")["minutes"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=2).mean()
    )

    # ---- opp_def_vs_pos: how hard this opponent defends against this position.
    # Grouped by (opponent, position_group), so the rate belongs to the defence
    # facing that matchup -- not to the player who happens to be in this row.
    games["_per36"] = games["points"] / games["minutes"].replace(0, np.nan) * PER_36

    # The rate belongs to the *defence* facing a matchup, so it pools every game
    # that matchup has seen. Two things that takes:
    #
    #  * the window must run in **date** order. `groupby` preserves frame order
    #    and this frame is sorted by player_id, so rolling in place would roll
    #    across interleaved players rather than across games.
    #  * it must exclude the whole **date**, not just the preceding row. Several
    #    players share one (opponent, position) on a given night, so `shift(1)`
    #    alone leaves same-night rows inside the window predicting that night --
    #    the fourth instance in this repo of a rolling window that leaks the game
    #    it is predicting (spec 14.1). Aggregating per date first and rolling
    #    over dates is what actually excludes it.
    games["_row"] = np.arange(len(games))
    per_date = (
        games.groupby(["opponent", "position_group", "game_date"], dropna=False)["_per36"]
        .mean()
        .rename("_date_rate")
        .reset_index()
        .sort_values(["opponent", "position_group", "game_date"], kind="mergesort")
    )
    per_date["_rate"] = per_date.groupby(["opponent", "position_group"])["_date_rate"].transform(
        lambda s: s.shift(1).rolling(window=window, min_periods=3).mean()
    )

    # Join back by carrying the original row number through, rather than by
    # re-merging on the three key columns: a multi-row-per-date key is exactly
    # where a join can hand back a neighbouring matchup's rate, and that is a
    # silently wrong feature rather than a crash.
    matched = per_date.merge(games[["_row", "opponent", "position_group", "game_date"]],
                             on=["opponent", "position_group", "game_date"], how="right")
    games["opp_def_vs_pos"] = (
        matched.sort_values("_row")["_rate"].to_numpy()
    )

    # An unmapped position has no defence-vs-position rate. NaN, not zero.
    games.loc[games["position_group"].isna(), "opp_def_vs_pos"] = np.nan

    return games.drop(columns=["_usage", "_usage_trend", "_per36", "_row"])