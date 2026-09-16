import numpy as np


def compute_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict]:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        bin_start, bin_end = bin_edges[i], bin_edges[i + 1]
        is_last_bin = i == n_bins - 1
        mask = (y_prob >= bin_start) & (y_prob < bin_end if not is_last_bin else y_prob <= bin_end)

        count = int(mask.sum())
        if count == 0:
            continue

        bins.append(
            {
                "bin_start": float(bin_start),
                "bin_end": float(bin_end),
                "predicted_rate": float(y_prob[mask].mean()),
                "actual_rate": float(y_true[mask].mean()),
                "count": count,
            }
        )

    return bins
