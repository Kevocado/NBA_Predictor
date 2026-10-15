"""The ship rule the comparison tools apply, as one tested function.

The merged AUC bar (docs/nba-calibration-decision-2026-10.md): AUC's 95% interval must EXCLUDE A DECLINE (lower bound
>= 0), not exclude zero. Log-loss, Brier, margin MAE and total MAE must each improve with an interval that excludes
zero in the better direction. The calibration gap must not exceed the absolute bar AND must not widen versus the
baseline measured on the same games.

"Must not widen" is sample-size-aware when the caller passes `gap_widened` (from `paired_gap_bootstrap`): on a
thousand-odd games the gap is a noisy max, so a point-estimate comparison accepts and rejects on noise. With
`gap_widened`, only a REAL widening (the paired interval of the difference lies above zero) fails the clause. The absolute
bar still applies either way, and with `gap_widened=None` the old strict point comparison is used unchanged.
"""
from __future__ import annotations

MAX_GAP_BAR = 0.0591
AUC_KEY = "win auc"


def ship_decision(boot: dict, gap: float, baseline_gap: float, bar: float = MAX_GAP_BAR,
                  gap_widened: bool | None = None) -> dict:
    """`boot` is {metric name -> paired_bootstrap result} with `improved` and `ci_low` (candidate minus baseline)."""
    if AUC_KEY not in boot:
        raise KeyError(f"{AUC_KEY!r} missing from the bootstrap results")
    auc_ok = boot[AUC_KEY]["ci_low"] >= 0
    others = [name for name in boot if name != AUC_KEY]
    missing = [name for name in others if not boot[name]["improved"]]
    not_wider = gap <= baseline_gap if gap_widened is None else not gap_widened
    gap_ok = gap <= bar and not_wider
    return {"auc_ok": auc_ok, "missing": missing, "gap_ok": gap_ok,
            "ships": bool(auc_ok and not missing and gap_ok)}
