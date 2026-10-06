"""Chronological holdout for the four prop regressors (spec section 7, gap G4).

The repo shipped prop models whose only evidence was **in-sample** metrics —
`ingest.train_player_prop_models` sets `training={"in_sample_metrics": True}` and
says so in the manifest. A model scored on the rows it was fitted against cannot
fail its own evaluation. This is the holdout that has never existed.

Two things this gets right that a naive version does not, both borrowed from
`pipeline/retrain.py` rather than reinvented:

* **Holdout rows are built from train+holdout, then selected.** Rolling features
  are shift(1), so a holdout player's first game must see its *training* games.
  Building features from the holdout slice alone gives every holdout row a
  truncated history that no serving scenario would have.
* **Features are built once**, over the concatenation, and the target is then
  picked per market. Building them inside a four-market loop recomputes the same
  frame four times and invites the four markets to disagree about the split.

A market whose target column is absent raises rather than reporting `inf`. An
`inf` MAE reads as a real, terrible number in the report, and the honest
statement is "this market had no target column", which is a different fact.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from nba_predictor.features.player_stats import build_player_feature_frame
from nba_predictor.models.evaluate.walk_forward import chronological_split

#: market name -> the column it actually lives in. "threes" is the market's
#: public name; the box score calls it fg3m. One map, resolved once, rather than
#: a chain of conditionals re-deriving it per row.
MARKET_TARGETS = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threes": "fg3m",
}

MARKETS = tuple(MARKET_TARGETS)


def prop_holdout_metrics(
    df: pd.DataFrame,
    factories: dict[str, Callable],
    holdout_fraction: float = 0.2,
) -> dict:
    """Holdout vs in-sample MAE per market on one chronological split.

    `df` is the player-game frame `ingest.to_player_training_frame` builds.
    `factories` maps market name to `(X, y) -> fitted model`.

    Every market is scored against the **same** split — one split, four targets.
    Four independent splits would let the four numbers describe four different
    periods and quietly stop being comparable to each other.
    """
    if not isinstance(df, pd.DataFrame) or len(df) < 2:
        raise ValueError("prop_holdout_metrics needs a player-game frame with at least 2 rows")
    if "game_date" not in df.columns:
        raise ValueError("player-game frame has no game_date; cannot split chronologically")

    train, holdout = chronological_split(
        df, date_col="game_date", holdout_fraction=holdout_fraction
    )

    # One feature build over train+holdout, so holdout rolling features inherit
    # the training history a real prediction would have (see module docstring).
    combined = pd.concat([train, holdout], ignore_index=True)
    combined_frame, feature_cols = build_player_feature_frame(combined)
    holdout_ids = set(holdout["game_id"])

    train_frame = combined_frame[~combined_frame["game_id"].isin(holdout_ids)]
    holdout_frame = combined_frame[combined_frame["game_id"].isin(holdout_ids)]

    if len(train_frame) == 0 or len(holdout_frame) == 0:
        raise ValueError(
            f"split left {len(train_frame)} train / {len(holdout_frame)} holdout rows; "
            "each player needs at least one prior game for the rolling features"
        )

    train_max = str(train["game_date"].max())
    holdout_min = str(holdout["game_date"].min())
    if train_max >= holdout_min:
        raise AssertionError(
            f"holdout is not strictly after train: {train_max} >= {holdout_min}"
        )

    results: dict[str, dict] = {}
    for market in MARKETS:
        if market not in factories:
            raise KeyError(f"no factory supplied for market {market!r}")
        target = MARKET_TARGETS[market]
        if target not in combined_frame.columns:
            raise KeyError(
                f"market {market!r} needs column {target!r}, which the frame does not have"
            )

        model = factories[market](train_frame[feature_cols], train_frame[target])
        in_sample = float(
            np.abs(np.asarray(model.predict(train_frame[feature_cols]), dtype=float) - train_frame[target]).mean()
        )
        holdout_pred = np.asarray(model.predict(holdout_frame[feature_cols]), dtype=float)
        holdout_mae = float(np.abs(holdout_pred - holdout_frame[target]).mean())

        results[market] = {
            "holdout_mae": holdout_mae,
            "in_sample_mae": in_sample,
            # The honest gap. Positive means the in-sample number was flattering,
            # which is the thing this whole task exists to measure.
            "generalisation_gap": holdout_mae - in_sample,
            "train_max_date": train_max,
            "holdout_min_date": holdout_min,
            "n_train": int(len(train_frame)),
            "n_holdout": int(len(holdout_frame)),
            "n_players": int(combined_frame["player_id"].nunique()),
            "feature_cols": list(feature_cols),
        }

    return results