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
    out = ship_decision(_boot(improved=False), gap=0.05, baseline_gap=0.06)
    assert set(out["missing"]) == {"win log_loss", "win brier", "margin mae", "total mae"} and not out["ships"]


def test_a_gap_under_the_bar_that_widened_versus_the_baseline_does_not_ship():
    out = ship_decision(_boot(), gap=0.04, baseline_gap=0.02)      # 0.04 < 0.0591 but doubled
    assert not out["gap_ok"] and not out["ships"]


def test_a_gap_over_the_absolute_bar_does_not_ship_even_if_it_narrowed():
    assert not ship_decision(_boot(), gap=0.07, baseline_gap=0.09)["ships"]
