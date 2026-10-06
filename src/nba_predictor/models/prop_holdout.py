import pandas as pd

from nba_predictor.features.player_stats import build_player_feature_frame
from nba_predictor.models.player_props import predict_player_stat


def prop_holdout_metrics(df: pd.DataFrame, factories: dict, holdout_fraction: float = 0.2) -> dict:
    """Chronological holdout evaluation for the 4 prop regressors.

    df must contain game_date and the 4 target columns: points, rebounds, assists, threes.
    factories maps market name to a function (X, y) -> fitted model.

    Returns metrics per market with holdout_mae, in_sample_mae, train_max_date,
    holdout_min_date, n_train, n_holdout.
    """
    from nba_predictor.models.evaluate.walk_forward import chronological_split

    markets = ["points", "rebounds", "assists", "threes"]

    # Use chronological split on the player-game frame as given; reuse existing helper
    train, holdout = chronological_split(df, date_col="game_date", holdout_fraction=holdout_fraction)

    result = {}
    for market in markets:
        target_col = market if market in df.columns else ("fg3m" if market == "threes" else market)
        # For threes, prefer fg3m as target if present in the training frame semantics
        if market == "threes" and "fg3m" in df.columns and target_col not in train.columns:
            target_col = "fg3m"
        elif market == "threes" and target_col not in train.columns:
            target_col = "threes"

        # Train model on train set
        train_frame, feature_cols = build_player_feature_frame(train.copy())
        holdout_frame, _ = build_player_feature_frame(holdout.copy())

        # Need to align features; if holdout has no rows after feature assembly, handle
        # But the requirement is to report honest numbers based on the split
        in_sample_mae = float("inf")
        holdout_mae = float("inf")
        train_max_date = train["game_date"].max() if len(train) > 0 else None
        holdout_min_date = holdout["game_date"].min() if len(holdout) > 0 else None

        if len(train_frame) > 0 and target_col in train_frame.columns:
            factory = factories[market]
            model = factory(train_frame[feature_cols], train_frame[target_col])
            preds_train = predict_player_stat(model, train_frame[feature_cols])
            in_sample_mae = float((preds_train - train_frame[target_col]).abs().mean())

        # Evaluate on holdout if possible
        if len(holdout_frame) > 0 and target_col in holdout_frame.columns:
            factory = factories[market]
            model = factory(train_frame[feature_cols] if len(train_frame) > 0 else pd.DataFrame(columns=feature_cols), train_frame[target_col] if len(train_frame) > 0 and target_col in train_frame.columns else pd.Series(dtype=float))
            # Refit properly
            model = factory(train_frame[feature_cols], train_frame[target_col])
            preds_hold = predict_player_stat(model, holdout_frame[feature_cols])
            holdout_mae = float((preds_hold - holdout_frame[target_col]).abs().mean())

        result[market] = {
            "holdout_mae": float(holdout_mae),
            "in_sample_mae": float(in_sample_mae),
            "train_max_date": train_max_date,
            "holdout_min_date": holdout_min_date,
            "n_train": int(len(train_frame)),
            "n_holdout": int(len(holdout_frame)),
        }

    return result
