"""The three arms of the carry-over A/B must be scored on the identical games.

The failure this exists to prevent is quiet. Two arms carry the same 783
predictions in a different row order; a positional restriction takes the first
758 of each, and the reliability table and paired bootstrap then compare game
i of one arm against game i of the other. Nothing raises, nothing looks wrong,
and the AUC lands wherever the mislabelling puts it -- which in this A/B was
below 0.5, under 0.47.

So alignment is by `game_id`, never by position, and an arm without ids is an
error rather than a fallback.
"""

import numpy as np
import pytest

from nba_predictor.models.evaluate.walk_forward_eval import (
    ids_of,
    phase_a_ids,
    restrict_to_ids,
    assert_same_games,
)

N = 12
GAME_IDS = [f"g{i:03d}" for i in range(N)]


def _classification_pooled(order, y, preds):
    """Walk-forward-metrics shape: ids, y, preds at the top level."""
    return {
        "y": [y[i] for i in order],
        "preds": [preds[i] for i in order],
        "game_ids": [GAME_IDS[i] for i in order],
        "n": len(order),
    }


def _carry_pooled(order, y, preds):
    """Walk-forward-carry-over shape: one sub-dict per target."""
    target = _classification_pooled(order, y, preds)
    return {
        "win": dict(target),
        "margin": dict(target),
        "total": {**target, "mae": 0.0},
    }


def _fixture():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, N)
    preds = np.clip(rng.beta(2, 2, N), 0.02, 0.98)
    return y, preds


def test_ids_of_reads_both_pooled_shapes():
    """The helper has to see ids whichever shape a walk-forward returned."""
    y, preds = _fixture()
    assert ids_of(_classification_pooled(range(N), y, preds)) == GAME_IDS
    assert ids_of(_carry_pooled(range(N), y, preds)) == GAME_IDS


def _carry_pooled_in_order(ids, y, preds):
    """Carry-over shape, ids given explicitly (no index shuffling of ids)."""
    target = {"y": list(y), "preds": list(preds), "game_ids": list(ids), "n": len(ids)}
    return {
        "win": dict(target),
        "margin": dict(target),
        "total": {**target, "mae": 0.0},
    }


def test_two_arms_in_different_orders_stay_paired_across_the_comparison():
    """The load-bearing one.

    This is how the A/B actually consumes the arms: `y` comes from one arm and
    `preds` from both, assembled positionally per target
    (`np.asarray(arm0["win"]["y"])`, `np.asarray(arm1["win"]["preds"])`).
    Row i of the three arrays must therefore be the same game in both arms.

    Under `[:n]` truncation each arm is cut to its own first n rows, and with
    the rows in a different order row i of the second arm is a *different game*.
    Nothing raises; the paired bootstrap and the reliability table just compare
    the wrong games against each other. That is what put win AUC at 0.47.
    """
    y, preds = _fixture()

    order_a = list(range(N))
    order_b = [7, 0, 11, 3, 5, 9, 1, 4, 10, 2, 8, 6]
    arm_a = _carry_pooled_in_order([GAME_IDS[i] for i in order_a], y, preds)
    arm_b_ids = [GAME_IDS[i] for i in order_b] + [f"x{i}" for i in range(4)]
    arm_b_y = [y[i] for i in order_b] + [0] * 4
    arm_b_p = [preds[i] for i in order_b] + [0.5] * 4
    arm_b = _carry_pooled_in_order(arm_b_ids, arm_b_y, arm_b_p)

    base = _classification_pooled(order_a, y, preds)
    keep_order = phase_a_ids(base, N)
    assert keep_order == GAME_IDS

    a = restrict_to_ids(arm_a, keep_order, N, "no carry-over")
    b = restrict_to_ids(arm_b, keep_order, N, "+ carry-over")
    assert_same_games(a, keep_order, N, "no carry-over")
    assert_same_games(b, keep_order, N, "+ carry-over")

    # What the tool does: three positional arrays, one per arm's target.
    y_a = a["win"]["y"]
    p_a = a["win"]["preds"]
    p_b = b["win"]["preds"]

    for i, gid in enumerate(a["win"]["game_ids"]):
        want = GAME_IDS.index(gid)
        assert p_a[i] == pytest.approx(preds[want]), (
            f"{gid}: arm A's own prediction is {p_a[i]}, expected {preds[want]}")
        assert p_b[i] == pytest.approx(preds[want]), (
            f"{gid}: arm B is holding another game's prediction ({p_b[i]}) at "
            f"row {i}; expected {preds[want]} for {gid}. The two arms are "
            "aligned by position, not by game_id."
        )


def test_an_arm_missing_phase_a_games_is_an_error_not_a_truncation():
    """A truncated arm must raise, not silently compare fewer games."""
    y, preds = _fixture()
    base = _classification_pooled(range(N), y, preds)
    keep_order = phase_a_ids(base, N)

    arm = _carry_pooled_in_order(GAME_IDS[:5], y[:5], preds[:5])
    with pytest.raises(ValueError, match="missing"):
        restrict_to_ids(arm, keep_order, N, "short arm")


def test_an_arm_without_game_ids_is_an_error():
    """No ids means no identity, so there is nothing to align and it must say so."""
    y, preds = _fixture()
    keep_order = GAME_IDS
    arm = {"win": {"y": list(y), "preds": list(preds), "n": N}, "margin": {}, "total": {}}
    with pytest.raises(ValueError, match="refusing to align by position"):
        restrict_to_ids(arm, keep_order, N, "no ids")


def test_a_baseline_without_game_ids_is_an_error():
    """The Phase A side must carry ids too; 'the first n rows' is not a fallback."""
    base = {"y": [0, 1] * 6, "preds": [0.4] * N, "n": N}
    with pytest.raises(ValueError, match="no game_ids"):
        phase_a_ids(base, N)


def test_assert_same_games_rejects_an_arm_with_extra_games():
    """783 games is not 758, even when it contains all of them."""
    y, preds = _fixture()
    base = _classification_pooled(range(N), y, preds)
    keep_order = phase_a_ids(base, N)

    arm = _carry_pooled_in_order(GAME_IDS + ["x0"], list(y) + [0], list(preds) + [0.5])
    with pytest.raises(ValueError, match="not aligned"):
        assert_same_games(arm, keep_order, N, "+ carry-over")
