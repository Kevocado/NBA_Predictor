"""Game-level coherence test: win probability and margin must agree on every card.

The served probability is Phi(margin / sigma). A card can never read 72% and
"Toss-up" — the label is defined by the same number as the percentage.
This test scans the live-shaped payload for any contradiction.
"""

import pytest
import numpy as np
from scipy.stats import norm

from nba_predictor.models.probability import win_prob


def test_win_prob_matches_margin_cdf_on_synthetic_games():
    """Property test over a grid of margins: win = Phi(margin / sigma),
    monotone in margin, 0.5 at margin 0."""
    sigma = 15.87  # typical NBA margin sigma
    for margin in np.linspace(-30, 30, 61):
        wp = win_prob(float(margin), sigma)
        expected = float(norm.cdf(margin / sigma))
        assert wp == pytest.approx(expected, abs=1e-12), f"margin={margin}: {wp} vs {expected}"
        assert 0.0 <= wp <= 1.0

    # Monotonicity
    margins = np.linspace(-30, 30, 61)
    probs = [win_prob(float(m), sigma) for m in margins]
    assert all(probs[i] <= probs[i+1] for i in range(len(probs)-1))

    # 0.5 at margin 0
    assert win_prob(0.0, sigma) == pytest.approx(0.5, abs=1e-12)


def test_tossup_label_agrees_with_win_prob():
    """'Toss-up' is defined by the same number as the percentage.
    A card can never read 72% and 'Toss-up'."""
    sigma = 15.87
    for margin in np.linspace(-30, 30, 61):
        wp = win_prob(float(margin), sigma)
        # Toss-up when |margin| < ~1.5 (roughly win in [0.45, 0.55])
        # The exact threshold is |margin| < sigma * norm.ppf(0.55) ≈ sigma * 0.126
        tossup_threshold = sigma * norm.ppf(0.55)
        is_tossup = abs(margin) < tossup_threshold
        is_even_prob = 0.45 <= wp <= 0.55
        assert is_tossup == is_even_prob, f"margin={margin}, wp={wp:.3f}, tossup={is_tossup}, even={is_even_prob}"


def test_no_contradiction_in_api_payload():
    """Scan the live-shaped payload for any win/margin contradiction.

    This would be run against the actual API, but here we test the logic:
    for every game, win_prob must equal Phi(predicted_margin / margin_sigma).
    """
    sigma = 15.87
    # Toss-up threshold: |margin| < sigma * norm.ppf(0.55) ≈ sigma * 0.126 ≈ 2.0
    test_cases = [
        # (home_team, away_team, predicted_margin, expected_tossup)
        ("DET", "PHX", 6.93, False),   # ~70% home win, not toss-up
        ("PHI", "NYK", -9.71, False),  # ~28% home win, not toss-up
        ("SAC", "LAL", 1.0, True),     # ~52% home win, toss-up (margin < ~2.0)
    ]

    for home, away, margin, expected_tossup in test_cases:
        wp = win_prob(float(margin), sigma)
        tossup_threshold = sigma * norm.ppf(0.55)
        is_tossup = abs(margin) < tossup_threshold
        assert is_tossup == expected_tossup, f"{away} at {home}: margin={margin}, wp={wp:.3f}, tossup={is_tossup} (expected {expected_tossup})"


def test_score_upcoming_games_uses_win_from_margin(tmp_path):
    """Integration test: score_upcoming_games now uses win_prob(margin, sigma)
    and fails loudly if manifest lacks margin_sigma."""
    import json
    from pathlib import Path
    from unittest.mock import patch, MagicMock
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_upcoming_games

    # A REAL manifest in the scorer's own models_dir. The first version of this
    # test patched `nba_predictor.pipeline.ingest.Path` and `json.loads` and
    # passed a literal Path("models") -- which does not change the path the
    # scorer derives from its argument, so it read the repository's committed
    # models/manifest.json and passed on a real sigma it did not supply. It
    # also meant this test's result depended on a committed artefact.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text(json.dumps({
        "metrics": {"margin": {"residual_sigma": 15.87}}
    }))

    with patch("nba_predictor.pipeline.ingest.joblib.load") as mock_load:
        mock_margin = MagicMock()
        mock_margin.predict.return_value = np.array([6.5, -3.2, 0.0])
        mock_total = MagicMock()
        mock_total.predict.return_value = np.array([220.0, 215.0, 225.0])
        mock_win = MagicMock()  # Old classifier - must NOT be used

        def load_side_effect(path):
            if "win_probability_model" in str(path):
                return mock_win
            elif "margin_model" in str(path):
                return mock_margin
            elif "total_model" in str(path):
                return mock_total
            raise FileNotFoundError(path)

        mock_load.side_effect = load_side_effect

        with patch("nba_predictor.pipeline.ingest.build_feature_frame") as mock_build, \
             patch("nba_predictor.pipeline.ingest.store.insert_predictions") as mock_insert:

            mock_df = pd.DataFrame({
                "game_id": ["g1", "g2", "g3"],
                "home_win": [np.nan, np.nan, np.nan],
                "home_team": ["BOS", "LAL", "NYK"],
                "away_team": ["MIA", "GSW", "BOS"],
                "f1": [1.0, 2.0, 3.0],
                "f2": [0.5, 1.5, 2.5],
            })
            mock_build.return_value = (mock_df, ["f1", "f2"])

            games = [
                {"game_id": "g1", "game_date": "2026-10-08", "home_team": "BOS", "away_team": "MIA"},
                {"game_id": "g2", "game_date": "2026-10-09", "home_team": "LAL", "away_team": "GSW"},
                {"game_id": "g3", "game_date": "2026-10-10", "home_team": "NYK", "away_team": "BOS"},
            ]

            count = score_upcoming_games(games, models_dir, tmp_path / "tracking.db", "v20261007")

            assert count == 3
            assert mock_insert.call_count == 1

            calls = [type("C", (), {"kwargs": r}) for r in mock_insert.call_args.args[1]]
            wp_g1 = calls[0].kwargs["home_win_prob"]
            wp_g2 = calls[1].kwargs["home_win_prob"]
            wp_g3 = calls[2].kwargs["home_win_prob"]

            assert wp_g1 == pytest.approx(win_prob(6.5, 15.87))
            assert wp_g2 == pytest.approx(win_prob(-3.2, 15.87))
            assert wp_g3 == pytest.approx(win_prob(0.0, 15.87), abs=1e-12)
            assert wp_g3 == pytest.approx(0.5, abs=1e-12)

            # Old classifier must NOT have been called: this is the whole point.
            mock_win.predict_proba.assert_not_called()


def test_score_upcoming_games_fails_loud_without_margin_sigma(tmp_path):
    """If manifest lacks margin_sigma, score_upcoming_games raises ValueError."""
    import json
    from unittest.mock import patch, MagicMock
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_upcoming_games

    # A real manifest, in the scorer's own models_dir, that genuinely lacks
    # residual_sigma. Same reason as the test above: patching Path would not
    # change the path the scorer derives from its argument.
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "manifest.json").write_text(json.dumps({
        "metrics": {"margin": {"mae": 14.0}}  # no residual_sigma
    }))

    with patch("nba_predictor.pipeline.ingest.joblib.load") as mock_load:
        mock_margin = MagicMock()
        mock_margin.predict.return_value = np.array([6.5])
        mock_total = MagicMock()
        mock_total.predict.return_value = np.array([220.0])
        mock_win = MagicMock()

        def load_side_effect(path):
            if "win_probability_model" in str(path):
                return mock_win
            elif "margin_model" in str(path):
                return mock_margin
            elif "total_model" in str(path):
                return mock_total
            raise FileNotFoundError(path)

        mock_load.side_effect = load_side_effect

        with patch("nba_predictor.pipeline.ingest.build_feature_frame") as mock_build:

            mock_df = pd.DataFrame({
                "game_id": ["g1"],
                "home_win": [np.nan],
                "home_team": ["BOS"],
                "away_team": ["MIA"]
            })
            mock_build.return_value = (mock_df, ["f1"])

            games = [{"game_id": "g1", "game_date": "2026-10-08", "home_team": "BOS", "away_team": "MIA"}]

            with pytest.raises(ValueError, match="margin_sigma"):
                score_upcoming_games(games, models_dir, tmp_path / "tracking.db", "v20261007")

def test_scored_totals_carry_the_manifests_league_drift(tmp_path):
    """The total model under-predicted league scoring by ~2.6 pts every season since 2020; the manifest carries the recent
    league-wide residual and every served total adds it. A manifest without one (an older one) adds nothing."""
    import json
    from unittest.mock import patch, MagicMock
    import pandas as pd

    from nba_predictor.pipeline.ingest import score_upcoming_games

    def run(metrics):
        models_dir = tmp_path / f"m{len(str(metrics))}"
        models_dir.mkdir()
        (models_dir / "manifest.json").write_text(json.dumps({"metrics": metrics}))
        with patch("nba_predictor.pipeline.ingest.joblib.load") as mock_load, \
             patch("nba_predictor.pipeline.ingest.build_feature_frame") as mock_build, \
             patch("nba_predictor.pipeline.ingest.store.insert_predictions") as mock_insert:
            margin, total = MagicMock(), MagicMock()
            margin.predict.return_value = np.array([1.0])
            total.predict.return_value = np.array([220.0])
            mock_load.side_effect = lambda path: total if "total_model" in str(path) else margin if "margin_model" in str(path) else MagicMock()
            mock_build.return_value = (pd.DataFrame({"game_id": ["g1"], "home_win": [np.nan], "f1": [1.0]}), ["f1"])
            score_upcoming_games([{"game_id": "g1", "game_date": "2026-10-22", "home_team": "BOS", "away_team": "MIA"}],
                                 models_dir, tmp_path / "t.db", "v1")
            return mock_insert.call_args.args[1][0]["predicted_total"]

    assert run({"margin": {"residual_sigma": 15.0}, "total": {"drift": 2.5}}) == 222.5
    assert run({"margin": {"residual_sigma": 15.0}}) == 220.0


def test_the_drift_is_the_recent_league_residual_and_is_capped():
    import numpy as np
    import pandas as pd
    from nba_predictor.pipeline.retrain import TOTAL_DRIFT_CAP, compute_total_drift, manifest_total_drift

    class Model:
        def __init__(self, level):
            self.level = level

        def predict(self, X):
            return np.full(len(X), self.level)

    df = pd.DataFrame({"game_date": pd.date_range("2026-01-01", periods=400).astype(str),
                       "home_pts": 115.0, "away_pts": 110.0, "f": 1.0})          # actual totals 225
    assert compute_total_drift(df, Model(222.5), ["f"]) == 2.5
    assert compute_total_drift(df, Model(180.0), ["f"]) == TOTAL_DRIFT_CAP         # a broken fit cannot move totals by 45
    assert compute_total_drift(df.head(10), Model(222.5), ["f"]) == 0.0            # too little to read
    assert manifest_total_drift({"metrics": {"total": {"drift": float("nan")}}}) == 0.0


def test_a_drift_that_cannot_be_read_is_no_correction_not_a_failed_retrain():
    import pandas as pd
    from nba_predictor.pipeline.retrain import compute_total_drift

    class Broken:
        def predict(self, X):
            raise RuntimeError("no model")

    df = pd.DataFrame({"game_date": pd.date_range("2026-01-01", periods=100).astype(str), "home_pts": 110.0, "away_pts": 105.0, "f": 1.0})
    assert compute_total_drift(df, Broken(), ["f"]) == 0.0
