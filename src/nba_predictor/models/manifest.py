import json
from pathlib import Path


def build_manifest(model_names: list[str], metrics: dict, model_version: str, trained_at: str) -> dict:
    return {
        "model_version": model_version,
        "trained_at": trained_at,
        "models": model_names,
        "metrics": metrics,
    }


def write_manifest(manifest: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


def append_manifest_history(manifest: dict, history_path: Path) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "a") as f:
        f.write(json.dumps(manifest) + "\n")
