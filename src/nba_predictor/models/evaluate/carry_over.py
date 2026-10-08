"""Fit the season carry-over weight in walk-forward, never in hindsight.

The weight is the one number a human is most tempted to pick by looking at the
answer. It is fitted here on each outer window's TRAINING data only, chosen from
a coarse grid by an inner time-ordered split, and it is scored against the
baseline on the outer window's test games -- which the search never sees.

The property under test is the ordering, not the accuracy: for outer window k,
the search may read nothing dated on or after that window's first test game.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_predictor.features.build import build_training_frame
from nba_predictor.features.four_factors import add_rolling_four_factors

#: Coarse grid on purpose. The carry-over weight has a flat optimum and a fine
#: grid would mostly be fitting the split's noise.
WEIGHT_GRID = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75)


def _long_form(games: pd.DataFrame) -> pd.DataFrame:
    """The long team-per-game frame the rolling features are built on."""
    cols = ["game_id", "game_date", "team", "fgm", "fga", "fg3m", "tov", "oreb", "opp_dreb", "fta"]

    def side(home: bool):
        s = "home" if home else "away"
        opp = "away" if home else "home"
        return games.rename(columns={
            f"{s}_team": "team", f"{s}_fgm": "fgm", f"{s}_fga": "fga", f"{s}_fg3m": "fg3m",
            f"{s}_tov": "tov", f"{s}_oreb": "oreb", f"{opp}_dreb": "opp_dreb", f"{s}_fta": "fta",
        })[cols]

    return pd.concat([side(True), side(False)], ignore_index=True)


def _features(games: pd.DataFrame, weight: float, window: int = 10) -> pd.DataFrame:
    """Home/away feature columns for `games`, with carry-over `weight` applied.

    Mirrors `build_feature_frame`: uses a merge on (game_id, team) rather than a
    simple map, because the rolling values are indexed by (game_id, team) and a
    plain map on team name alone cannot distinguish the two teams in the same game.
    """
    long = add_rolling_four_factors(
        _long_form(games), window=window, carry_over_weight_value=weight
    )
    factors = ["efg_pct", "tov_rate", "orb_pct", "ft_rate"]
    rolled = long.set_index(["game_id", "team"])[[f"{f}_roll" for f in factors]].reset_index()

    out = games[["game_id", "game_date", "home_team", "away_team"]].copy()
    out["margin"] = games["home_pts"] - games["away_pts"]
    for side, team_col in (("home", "home_team"), ("away", "away_team")):
        merged = out.merge(
            rolled[["game_id", "team"] + [f"{f}_roll" for f in factors]],
            left_on=["game_id", team_col] if side == "home" else ["game_id", team_col],
            right_on=["game_id", "team"],
            how="left",
        )
        for factor in factors:
            out[f"{side}_{factor}_roll"] = merged[f"{factor}_roll"].to_numpy()
    return out


FEATURE_COLS = [
    f"{side}_{factor}_roll"
    for side in ("home", "away")
    for factor in ("efg_pct", "tov_rate", "orb_pct", "ft_rate")
]


#: Fraction of the TRAINING slice, by distinct date, held back as the inner
#: validation set. The rest fits. Time-ordered, and derived from the training
#: slice alone.
INNER_HOLDOUT_FRACTION = 0.3


def _inner_split(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Split the training slice into an inner fit and an inner validation set.

    **The split point comes from the training slice, not from the outer cutoff.**
    An earlier version of this reused the outer window's first test date as the
    inner boundary, which made the validation half empty every time -- `train` is
    already everything before that date -- so every grid point scored NaN and the
    search silently reported "unusable" for all of them.
    """
    dates = np.sort(pd.unique(train["game_date"]))
    if len(dates) < 4:
        return train.iloc[:0], train.iloc[:0], ""
    cut = str(dates[max(1, int(len(dates) * (1 - INNER_HOLDOUT_FRACTION)))])
    return train[train.game_date < cut], train[train.game_date >= cut], cut


def _fit_mae(train: pd.DataFrame, weight: float, cutoff: str, window: int = 10) -> float:
    """Inner score: MAE on the held-back tail of the TRAINING slice.

    `cutoff` is the outer window's first test date and is used only to drop
    anything at or after it -- a belt-and-braces filter, since `train` is
    already strictly before it.
    """
    from sklearn.linear_model import Ridge

    train = train[train.game_date < cutoff]
    early, late, inner_cut = _inner_split(train)
    if len(early) < 50 or len(late) < 20 or not inner_cut:
        return float("nan")

    feats = _features(early, weight, window=window)
    target = feats["margin"].to_numpy()
    X = feats[FEATURE_COLS].fillna(0.0).to_numpy()
    if np.isnan(target).any() or np.isnan(X).any():
        return float("nan")

    model = Ridge(alpha=1.0).fit(X, target)

    # Features for the WHOLE training slice, then score only the inner
    # validation tail. Building them from `late` alone would give those games no
    # rolling history, which is not what the model will see at prediction time.
    scored = _features(train, weight, window=window)
    scored = scored[scored.game_date >= inner_cut]
    if not len(scored):
        return float("nan")
    preds = model.predict(scored[FEATURE_COLS].fillna(0.0).to_numpy())
    return float(np.mean(np.abs(preds - scored["margin"].to_numpy())))


def fit_carry_over_weight(
    train: pd.DataFrame,
    *,
    cutoff: str,
    grid=WEIGHT_GRID,
    window: int = 10,
) -> dict:
    """Pick the carry-over weight for one outer window from its training data.

    `cutoff` is the outer window's FIRST TEST GAME date. Nothing on or after it
    is read, so the search cannot see the games it will be scored on.

    Returns the chosen weight plus every grid point's inner score, because a
    weight chosen without its alternatives visible is not reviewable -- a flat
    optimum and a sharp one mean very different things about how much to trust
    it.
    """
    scores = {w: _fit_mae(train, w, cutoff, window=window) for w in grid}
    usable = {w: s for w, s in scores.items() if np.isfinite(s)}
    if not usable:
        return {"weight": 0.0, "scores": scores, "usable": False,
                "reason": "no grid point could be scored on this training slice"}
    best = min(usable, key=usable.get)
    return {"weight": best, "scores": scores, "usable": True,
            "best_score": usable[best], "worst_score": max(usable.values())}


def walk_forward_carry_over(
    current: pd.DataFrame,
    history: pd.DataFrame | None,
    model_factories: dict,
    *,
    windows: int = 4,
    date_col: str = "game_date",
    target_col: str = "home_win",
    grid=WEIGHT_GRID,
    feature_builder=None,
) -> dict:
    """Walk-forward where the carry-over weight is fitted per window.

    `model_factories` maps a target name ("win", "margin", "total") to a
    `factory(train_df, feature_cols) -> predictor`. The predictor is handed the
    test frame and picks its own columns, matching `walk_forward_metrics`.

    For each outer window: fit the weight on that window's TRAINING data, build
    the feature frame for it, train on the window's training rows and score its
    test rows. **The fitted weight never sees the window it scores** -- it is
    chosen from `grid` by an inner time-ordered split inside the training slice.

    `feature_builder(games, weight) -> (frame, feature_cols)` is injected so a
    test can count the calls; the default builds the production frame.

    Returns per-window records (including the fitted weight and every grid
    score) and pooled per-game predictions and outcomes for each target.
    """
    from sklearn.metrics import mean_absolute_error

    from nba_predictor.features.build import build_training_frame
    from nba_predictor.models.evaluate.walk_forward_eval import expanding_windows

    if feature_builder is None:
        def feature_builder(games, weight):
            return build_training_frame(games, carry_over_weight=weight)

    ordered = current.sort_values(date_col).reset_index(drop=True)
    hist = None
    if history is not None and len(history):
        hist = history.sort_values(date_col).reset_index(drop=True)

    records: list[dict] = []
    # Map the factory's target name to the column name in the frame.
    target_to_col = {t: ("home_margin" if t == "margin" else "home_total" if t == "total" else target_col)
                     for t in model_factories}
    collected: dict[str, dict[str, list]] = {
        t: {"preds": [], "y": [], "game_ids": []} for t in model_factories
    }

    for i, (train_idx, test_idx) in enumerate(expanding_windows(ordered[date_col], windows)):
        cutoff = str(ordered.iloc[test_idx][date_col].min())
        raw_train = ordered.iloc[train_idx]
        if hist is not None:
            extra = hist[hist[date_col].astype(str) < cutoff]
            if len(extra):
                raw_train = pd.concat([extra, raw_train], ignore_index=True)
        raw_train = raw_train.sort_values(date_col).reset_index(drop=True)

        train_max = str(raw_train[date_col].max())
        if train_max >= cutoff:
            raise AssertionError(
                f"window {i}: train_max_date {train_max} >= test_min_date {cutoff}"
            )

        fit = fit_carry_over_weight(raw_train, cutoff=cutoff, grid=grid)
        weight = fit["weight"]

        # CRITICAL: the feature frame must include history so carry-over has prior
        # season data to regress toward. Build it over the FULL combined frame
        # (history + ordered), so that rolling features have their full context.
        # Then split by game_ids so that ONLY test games are in test_df.
        full = pd.concat([hist, ordered], ignore_index=True) if hist is not None and len(hist) else ordered
        frame, feature_cols = feature_builder(full, weight)
        frame = frame.assign(
            home_margin=full["home_pts"] - full["away_pts"],
            home_total=full["home_pts"] + full["away_pts"],
            home_win=(full["home_pts"] > full["away_pts"]).astype(int),
        )
        test_ids = set(ordered.iloc[test_idx]["game_id"])
        train_df = frame[
            (frame[date_col].astype(str) < cutoff) & ~frame["game_id"].isin(test_ids)
        ].reset_index(drop=True)
        test_df = frame[frame["game_id"].isin(test_ids)].reset_index(drop=True)
        if len(train_df) and str(train_df[date_col].max()) >= cutoff:
            raise AssertionError(f"window {i}: outer train contains games on/after {cutoff}")
        assert len(test_df) == len(test_idx), (
            f"window {i}: feature frame has {len(test_df)} test rows but the "
            f"window has {len(test_idx)}"
        )

        preds: dict[str, np.ndarray] = {}
        for target, factory in model_factories.items():
            predictor = factory(train_df, feature_cols)
            if target == "win":
                preds[target] = np.asarray(predictor(test_df), dtype=float)
                collected[target]["y"].append(test_df[target_col].to_numpy())
            else:
                preds[target] = np.asarray(predictor(test_df), dtype=float)
                y_col = target_to_col[target]
                collected[target]["y"].append(test_df[y_col].to_numpy())
            collected[target]["preds"].append(preds[target])
            collected[target]["game_ids"].append(test_df["game_id"].to_numpy())

        records.append({
            "window": i,
            "cutoff": cutoff,
            "weight": weight,
            "usable": fit["usable"],
            "inner_scores": {str(k): v for k, v in fit["scores"].items()},
            "n_train": len(train_df),
            "n_test": len(test_df),
        })

    pooled: dict[str, dict] = {}
    for target, bag in collected.items():
        y = np.concatenate(bag["y"])
        p = np.concatenate(bag["preds"])
        gids = np.concatenate(bag["game_ids"])
        entry = {"y": y.tolist(), "preds": p.tolist(), "game_ids": gids.tolist(), "n": int(len(y))}
        if target == "win":
            from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
            entry.update({
                "log_loss": float(log_loss(y, p, labels=[0, 1])),
                "brier": float(brier_score_loss(y, p)),
                "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            })
        else:
            entry["mae"] = float(mean_absolute_error(y, p))
        pooled[target] = entry

    return {"windows": records, "pooled": pooled}
