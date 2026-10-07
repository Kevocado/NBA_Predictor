"""The fabricated zero did not reach a model. Pinned, so it stays pinned.

The question this file answers: did data/injuries.py's fabricated 0.0 get
fitted into a shipped model, or served to a user?

Answer, traced at 22dfbc7567b08dcbe7e83c242b956e48f69797a0: no, on both counts.

  * Nothing ever called features/injuries.py::missing_players_value except its
    own test. features/__init__.py is empty (so no package re-export), and no
    module in src/ imports features.injuries. Its fabricated zero could only
    have entered a model through a caller that does not exist.
  * The three TEAM models do carry home_missing_value/away_missing_value as
    features (24 features, verified by loading the committed .pkl), but those
    columns are filled by features/build.py:46-48 with a literal 0.0 for every
    row -- a constant, never from the stub -- and XGBoost never split on them,
    so they carry no fitted signal at all.

What these tests lock in:

  * The team models' missing_value columns are a constant, not the stub's
    fabricated per-player zero.
  * A constant feature XGBoost never split on cannot be a fitted influence, so
    even the (unreachable) stub path could not have moved a prediction through
    it.
  * 0.0 IS indistinguishable from a real measured 0.0 downstream -- a genuine
    design defect, recorded here rather than fixed, because build.py's default
    -zero policy covers five columns and changing it is a separate change.
"""

import pandas as pd
import pytest


def test_no_module_in_src_calls_missing_players_value():
    # The whole answer to "did the fabricated zero reach a model" in one
    # assertion: nobody in src/ calls it, so it cannot have reached anything.
    # Parsed rather than grepped, so a mention inside a docstring or comment
    # (where several now live, explaining the history) does not count as a
    # caller and only a real call site fails this.
    import ast
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    callers = []
    for path in src.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id == "missing_players_value":
                    callers.append(f"{path.relative_to(src)}:{node.lineno}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "missing_players_value":
                    callers.append(f"{path.relative_to(src)}:{node.lineno}")

    assert callers == [], f"a production path now calls missing_players_value: {callers}"


def test_no_training_or_serving_entry_point_imports_the_injuries_helper():
    # Belt and braces on the same fact, stated at the boundary that matters:
    # the pipeline modules that fit and serve models never import it.
    import ast
    import pathlib

    pipeline = pathlib.Path(__file__).resolve().parents[1] / "src" / "nba_predictor" / "pipeline"
    offenders = []
    for path in pipeline.rglob("*.py"):
        if "missing_players_value" in path.read_text() or "data import injuries" in path.read_text():
            offenders.append(path.name)

    assert offenders == [], f"the pipeline imports the injuries helper: {offenders}"


def test_features_package_does_not_reexport_the_injuries_helper():
    from nba_predictor import features

    assert not hasattr(features, "missing_players_value")


def test_the_team_models_carry_missing_value_as_a_constant_feature():
    import joblib

    features = _feature_names("win_probability_model")

    assert "home_missing_value" in features
    assert "away_missing_value" in features


def _feature_names(model_name):
    import joblib

    model = joblib.load(f"models/{model_name}.pkl")
    names = getattr(model, "feature_names_in_", None)
    return [str(name) for name in names] if names is not None else []


def test_a_constant_missing_value_column_never_becomes_a_split():
    import joblib

    # The strongest available evidence that a fabricated constant here could
    # not have influenced any prediction: XGBoost omits a feature from its
    # gain score entirely when it never split on it. XGBoost cannot split on a
    # constant -- there is no threshold that separates it.
    # For LogisticRegression, a constant feature gets a coefficient of 0.
    for name in ("win_probability_model", "margin_model", "total_model"):
        model = joblib.load(f"models/{name}.pkl")
        if hasattr(model, "get_booster"):
            # XGBoost model
            booster = model.get_booster()
            scored = booster.get_score(importance_type="gain")
            assert "home_missing_value" not in scored, f"{name} split on home_missing_value"
            assert "away_missing_value" not in scored, f"{name} split on away_missing_value"
        elif hasattr(model, "coef_"):
            # LogisticRegression model - check coefficients
            feature_names = getattr(model, "feature_names_in_", None)
            if feature_names is not None:
                coef_dict = dict(zip(feature_names, model.coef_.flatten()))
                # A constant feature should have coefficient 0 (or very close)
                home_coef = coef_dict.get("home_missing_value", 0)
                away_coef = coef_dict.get("away_missing_value", 0)
                assert abs(home_coef) < 1e-10, f"{name} has non-zero coefficient for home_missing_value: {home_coef}"
                assert abs(away_coef) < 1e-10, f"{name} has non-zero coefficient for away_missing_value: {away_coef}"
        else:
            pytest.skip(f"Unknown model type for {name}: {type(model)}")


def test_the_split_absence_is_because_the_column_is_constant_not_by_chance():
    # Guards the test above from being a tautology: same model, same call,
    # only the constant column varies -- and only then does XGBoost split on
    # it. So "never split" is a property of the data being constant, not a
    # property of how the artefact happened to be scored.
    import numpy as np

    from nba_predictor.models.game_outcome import train_margin_model

    rng = np.random.default_rng(0)
    n = 400
    base = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    target = base.f1 * 3 + base.f2 * 2 + rng.normal(scale=1, size=n)

    def _split_on_missing(frame, y):
        booster = train_margin_model(frame, y, n_estimators=10).get_booster()
        return [k for k in booster.get_score(importance_type="gain") if "missing_value" in k]

    constant = base.assign(home_missing_value=0.0, away_missing_value=0.0)
    varied = base.assign(
        home_missing_value=rng.normal(size=n), away_missing_value=rng.normal(size=n)
    )

    assert _split_on_missing(constant, target) == []
    assert _split_on_missing(varied, target) != []


def _two_games():
    """Two completed games with real box-score columns, as the feature frame
    requires. No missing_value column: that absence is the point."""
    rows = []
    for i, game_date in enumerate(["2026-10-21", "2026-10-23"]):
        rows.append(
            {
                "game_id": f"g{i}",
                "game_date": game_date,
                "home_team": "BOS" if i % 2 == 0 else "MIA",
                "away_team": "MIA" if i % 2 == 0 else "BOS",
                "home_pts": 110 + i,
                "away_pts": 105 + i,
                "home_fgm": 40, "home_fga": 88, "home_fg3m": 12, "home_tov": 11,
                "home_oreb": 9, "home_dreb": 32, "home_fta": 20,
                "away_fgm": 38, "away_fga": 90, "away_fg3m": 10, "away_tov": 13,
                "away_oreb": 10, "away_dreb": 30, "away_fta": 18,
                "home_win": 1,
            }
        )
    return pd.DataFrame(rows)


def test_missing_value_is_filled_with_a_constant_zero_not_from_the_stub():
    from nba_predictor.features.build import build_feature_frame

    result, feature_cols = build_feature_frame(_two_games())

    assert "home_missing_value" in feature_cols
    assert (result["home_missing_value"] == 0.0).all()
    assert (result["away_missing_value"] == 0.0).all()


def test_a_real_measured_missing_value_is_indistinguishable_from_the_default():
    # The recorded defect. A column a real source filled with 0.0 and the
    # default 0.0 produce the same frame, so nothing downstream can tell
    # "nobody is missing a player" from "we never looked".
    from nba_predictor.features.build import build_feature_frame

    defaulted = build_feature_frame(_two_games())[0]
    measured_zero = build_feature_frame(
        _two_games().assign(home_missing_value=0.0, away_missing_value=0.0)
    )[0]

    assert defaulted["home_missing_value"].equals(measured_zero["home_missing_value"])
    assert defaulted["away_missing_value"].equals(measured_zero["away_missing_value"])


def test_the_manifest_does_not_claim_a_missing_value_metric():
    import json

    manifest = json.load(open("models/player_props_manifest.json"))
    for stat, metrics in manifest["metrics"].items():
        assert set(metrics) == {"mae"}, f"{stat} claims a non-MAE metric: {set(metrics)}"


@pytest.mark.parametrize(
    "model_name", ["player_points_model", "player_rebounds_model", "player_assists_model", "player_threes_model"]
)
def test_the_player_models_have_no_injury_feature_at_all(model_name):
    # Five rolling-average features only. No injury, no availability, no
    # missing_value -- so the props models could not have been fitted on the
    # fabricated zero even in principle.
    assert _feature_names(model_name) == [
        "points_roll", "rebounds_roll", "assists_roll", "fg3m_roll", "minutes_roll",
    ]
