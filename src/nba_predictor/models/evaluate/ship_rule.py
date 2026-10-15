"""The ship rule the comparison tools apply, as one tested function.

The merged AUC bar (docs/nba-calibration-decision-2026-10.md): AUC's 95% interval must EXCLUDE A DECLINE (lower bound
>= 0), not exclude zero. Log-loss, Brier, margin MAE and total MAE must each improve with an interval that excludes
zero in the better direction. The calibration gap must not exceed the absolute bar AND must not widen versus the
baseline measured on the same games.

"Must not widen" is sample-size-aware when the caller passes `gap_widened` (from `paired_gap_bootstrap`): on a
thousand-odd games the gap is a noisy max, so a point-estimate comparison accepts and rejects on noise. With
`gap_widened`, only a REAL widening (the paired interval of the difference lies above zero) fails the clause. The absolute
bar still applies either way, and with `gap_widened=None` the old strict point comparison is used unchanged.

Promotion mode (2026-10-15, Kevin): the default is ANY-IMPROVES-NONE-WORSE. A candidate ships when at least one metric
improves with an interval excluding zero (any of win log-loss / Brier / AUC, margin MAE, total MAE) AND no metric is
significantly WORSE (its interval lies wholly on the worse side). A metric that merely stays level no longer blocks a
promotion that clearly helps another target. AUC must still exclude a decline, and the calibration clause above is
unchanged. `mode="all_improve"` keeps the previous rule (every non-AUC metric must improve) for callers that want it.
"""
from __future__ import annotations

MAX_GAP_BAR = 0.0591
AUC_KEY = "win auc"


def _significantly_worse(name: str, result: dict) -> bool:
    """The interval of (candidate - baseline) lies wholly on the worse side. Uses `lower_is_better` when present;
    otherwise AUC is higher-is-better and every other metric is lower-is-better."""
    if result.get("lower_is_better", name != AUC_KEY):
        return result.get("ci_low", float("-inf")) > 0
    return result.get("ci_high", float("inf")) < 0


def ship_decision(boot: dict, gap: float, baseline_gap: float, bar: float = MAX_GAP_BAR,
                  gap_widened: bool | None = None, mode: str = "any_improves_none_worse") -> dict:
    """`boot` is {metric name -> paired_bootstrap result} with `improved`, `ci_low`, `ci_high`, `lower_is_better`
    (candidate minus baseline)."""
    if mode not in ("any_improves_none_worse", "all_improve"):
        raise ValueError(f"unknown mode {mode!r}")
    if AUC_KEY not in boot:
        raise KeyError(f"{AUC_KEY!r} missing from the bootstrap results")
    auc_ok = boot[AUC_KEY]["ci_low"] >= 0
    others = [name for name in boot if name != AUC_KEY]
    missing = [name for name in others if not boot[name]["improved"]]
    not_wider = gap <= baseline_gap if gap_widened is None else not gap_widened
    gap_ok = gap <= bar and not_wider
    worse = [name for name in boot if _significantly_worse(name, boot[name])]
    improved = [name for name in boot if boot[name].get("improved")]
    if mode == "all_improve":
        metrics_ok = not missing
    else:
        metrics_ok = bool(improved) and not worse
    return {"auc_ok": auc_ok, "missing": missing, "worse": worse, "improved": improved, "gap_ok": gap_ok,
            "ships": bool(auc_ok and metrics_ok and gap_ok)}
