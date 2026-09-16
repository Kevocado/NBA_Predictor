import json


def test_build_manifest_shape():
    from nba_predictor.models.manifest import build_manifest

    manifest = build_manifest(
        model_names=["win_probability", "margin", "total"],
        metrics={"win_probability": {"accuracy": 0.65}},
        model_version="v1",
        trained_at="2026-11-01T00:00:00",
    )

    assert manifest["model_version"] == "v1"
    assert manifest["trained_at"] == "2026-11-01T00:00:00"
    assert manifest["models"] == ["win_probability", "margin", "total"]
    assert manifest["metrics"]["win_probability"]["accuracy"] == 0.65


def test_write_manifest_creates_valid_json(tmp_path):
    from nba_predictor.models.manifest import build_manifest, write_manifest

    manifest = build_manifest(["win_probability"], {}, "v1", "2026-11-01T00:00:00")
    path = tmp_path / "manifest.json"
    write_manifest(manifest, path)

    with open(path) as f:
        loaded = json.load(f)
    assert loaded == manifest


def test_append_manifest_history_adds_one_line_per_call(tmp_path):
    from nba_predictor.models.manifest import append_manifest_history, build_manifest

    history_path = tmp_path / "manifest_history.jsonl"
    manifest_v1 = build_manifest(["win_probability"], {}, "v1", "2026-11-01T00:00:00")
    manifest_v2 = build_manifest(["win_probability"], {}, "v2", "2026-11-02T00:00:00")

    append_manifest_history(manifest_v1, history_path)
    append_manifest_history(manifest_v2, history_path)

    lines = history_path.read_text().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["model_version"] == "v1"
    assert json.loads(lines[1])["model_version"] == "v2"
