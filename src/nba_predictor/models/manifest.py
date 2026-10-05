import json
from pathlib import Path


def build_manifest(
    model_names: list[str] | None = None,
    metrics: dict | None = None,
    model_version: str = "v1",
    trained_at: str = "",
    training: dict | None = None,
    **kwargs,
) -> dict:
    manifest = {
        "model_version": model_version,
        "trained_at": trained_at,
        "models": model_names or [],
        "metrics": metrics or {},
        "training": training or {},
    }
    # Merge any additional top-level keys
    for k, v in kwargs.items():
        if k not in manifest:
            manifest[k] = v
    # Also merge metrics keys as top-level if explicitly requested? test passes metrics dict with log_loss etc
    # but expects top-level in manifest per test case? check the test: test_manifest_emits_probability_metrics
    # does: build_manifest(metrics={"log_loss": 0.63, "brier": 0.22, "auc": 0.60}) and asserts top-level
    m = metrics or {}
    for k in ("log_loss", "brier", "auc", "margin_mae", "total_mae", "mae", "naive_mae"):
        if k in m and k not in manifest:
            manifest[k] = m[k]
    return manifest


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def append_manifest_history(manifest: dict, history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(manifest) + "\n")
