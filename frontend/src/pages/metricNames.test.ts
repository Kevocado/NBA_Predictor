import { describe, expect, it } from "vitest";
import { METRIC_NAMES } from "./ModelSummaryPage";

/**
 * The metric keys the backend actually emits.
 *
 * Read off `pipeline/retrain.py`, not off the manifest on disk: the shipped
 * manifest is the pre-Task-4 artifact and carries only `accuracy` and `mae`, so
 * it would happily certify a label map that has since drifted in both
 * directions. This is a transcription and it is the thing under test — when
 * retrain.py gains or drops a metric, this list has to change with it, and that
 * is the moment the label map is wrong.
 */
const EMITTED = [
  "accuracy",
  "log_loss",
  "brier",
  "auc",
  "mae",
  "wf_mae",
  "wf_naive_mae_fixed",
  "residual_sigma",
];

describe("METRIC_NAMES", () => {
  it("labels every metric the backend emits", () => {
    // No fallback: an unlabelled metric renders as its raw snake_case key, so
    // the page shows "wf_naive_mae_fixed 16.375" and calls it a day.
    const unlabelled = EMITTED.filter((key) => !(key in METRIC_NAMES));
    expect(unlabelled).toEqual([]);
  });

  it("carries no label the backend never emits", () => {
    // `brier_score` and `rmse` were here for metrics nothing produces. They are
    // dead weight that reads as coverage: someone auditing the page would
    // conclude Brier and RMSE are reported, and they are not.
    const dead = Object.keys(METRIC_NAMES).filter((key) => !EMITTED.includes(key));
    expect(dead).toEqual([]);
  });

  it("distinguishes the holdout MAE from the walk-forward one", () => {
    // Two different numbers under one label is how a reader ends up comparing
    // the published 16.512 against the naive 16.375 thinking they are the same
    // measurement. `mae` is the holdout; `wf_mae` is pooled out-of-fold.
    expect(METRIC_NAMES.mae).not.toBe(METRIC_NAMES.wf_mae);
  });

  it("names the naive baseline for what it is", () => {
    expect(METRIC_NAMES.wf_naive_mae_fixed.toLowerCase()).toContain("naive");
  });
});
