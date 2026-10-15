from nba_predictor.models.evaluate.ship_rule import ship_decision


def _boot(auc_low=0.0, improved=True):
    base = {"improved": improved, "ci_low": -0.01}
    return {"win log_loss": dict(base), "win brier": dict(base), "win auc": {"improved": False, "ci_low": auc_low},
            "margin mae": dict(base), "total mae": dict(base)}


def test_auc_only_has_to_exclude_a_decline_not_show_a_gain():
    assert ship_decision(_boot(auc_low=0.0), gap=0.05, baseline_gap=0.05)["ships"]
    assert ship_decision(_boot(auc_low=0.004), gap=0.05, baseline_gap=0.06)["ships"]


def test_an_auc_interval_that_includes_a_decline_does_not_ship():
    out = ship_decision(_boot(auc_low=-0.0001), gap=0.05, baseline_gap=0.06)
    assert not out["auc_ok"] and not out["ships"]


def test_the_other_four_metrics_must_each_improve():
    out = ship_decision(_boot(improved=False), gap=0.05, baseline_gap=0.06, mode="all_improve")
    assert set(out["missing"]) == {"win log_loss", "win brier", "margin mae", "total mae"} and not out["ships"]


def test_a_gap_under_the_bar_that_widened_versus_the_baseline_does_not_ship():
    out = ship_decision(_boot(), gap=0.04, baseline_gap=0.02)      # 0.04 < 0.0591 but doubled
    assert not out["gap_ok"] and not out["ships"]


def test_a_gap_over_the_absolute_bar_does_not_ship_even_if_it_narrowed():
    assert not ship_decision(_boot(), gap=0.07, baseline_gap=0.09)["ships"]


def test_a_noisy_widening_passes_when_the_paired_interval_says_it_is_not_real():
    # the point estimate widened 0.02 -> 0.04, but the bootstrap interval of the difference includes zero
    out = ship_decision(_boot(), gap=0.04, baseline_gap=0.02, gap_widened=False)
    assert out["gap_ok"] and out["ships"]


def test_a_real_widening_still_fails_even_under_the_bar():
    out = ship_decision(_boot(), gap=0.04, baseline_gap=0.02, gap_widened=True)
    assert not out["gap_ok"] and not out["ships"]


def test_the_absolute_bar_is_never_waived_by_the_sample_size_aware_clause():
    assert not ship_decision(_boot(), gap=0.07, baseline_gap=0.09, gap_widened=False)["ships"]


def test_without_gap_widened_the_old_strict_comparison_is_unchanged():
    assert not ship_decision(_boot(), gap=0.04, baseline_gap=0.02)["ships"]


def _mixed(**changes):
    """win auc ok; margin MAE improved; everything else level (interval spans zero)."""
    level = {"improved": False, "ci_low": -0.01, "ci_high": 0.01, "lower_is_better": True}
    boot = {"win log_loss": dict(level), "win brier": dict(level),
            "win auc": {"improved": False, "ci_low": 0.0, "ci_high": 0.02, "lower_is_better": False},
            "margin mae": {"improved": True, "ci_low": -1.1, "ci_high": -0.6, "lower_is_better": True},
            "total mae": dict(level)}
    for k, v in changes.items():
        boot[k.replace("_", " ")] = {**boot[k.replace("_", " ")], **v}
    return boot


def test_improving_one_target_while_the_others_stay_level_now_ships():
    out = ship_decision(_mixed(), gap=0.04, baseline_gap=0.04)
    assert out["improved"] == ["margin mae"] and out["worse"] == [] and out["ships"]


def test_improving_one_target_but_significantly_hurting_another_does_not_ship():
    out = ship_decision(_mixed(total_mae={"ci_low": 0.2, "ci_high": 0.9}), gap=0.04, baseline_gap=0.04)   # total MAE reliably worse
    assert out["worse"] == ["total mae"] and not out["ships"]


def test_nothing_improving_does_not_ship_even_if_nothing_is_worse():
    flat = _mixed(margin_mae={"improved": False, "ci_low": -0.2, "ci_high": 0.2})
    assert not ship_decision(flat, gap=0.04, baseline_gap=0.04)["ships"]


def test_a_worse_auc_interval_that_includes_a_decline_still_blocks_a_margin_win():
    out = ship_decision(_mixed(win_auc={"ci_low": -0.01, "ci_high": 0.02}), gap=0.04, baseline_gap=0.04)
    assert not out["auc_ok"] and not out["ships"]


def test_the_calibration_clause_still_applies_to_a_promotion_on_one_target():
    assert not ship_decision(_mixed(), gap=0.07, baseline_gap=0.09)["ships"]            # over the absolute bar
    assert not ship_decision(_mixed(), gap=0.04, baseline_gap=0.02, gap_widened=True)["ships"]


def test_all_improve_mode_keeps_the_previous_rule():
    assert not ship_decision(_mixed(), gap=0.04, baseline_gap=0.04, mode="all_improve")["ships"]
