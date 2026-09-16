import pandas as pd


def chronological_split(df: pd.DataFrame, date_col: str, holdout_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(df) < 2:
        raise ValueError("Need at least 2 rows to perform a chronological split")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    holdout_size = max(1, int(len(sorted_df) * holdout_fraction))
    split_index = len(sorted_df) - holdout_size

    train = sorted_df.iloc[:split_index].reset_index(drop=True)
    holdout = sorted_df.iloc[split_index:].reset_index(drop=True)
    return train, holdout
