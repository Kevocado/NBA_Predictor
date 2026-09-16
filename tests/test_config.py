# tests/test_config.py
import os
from pathlib import Path

import pytest


def test_project_root_is_repo_root():
    from nba_predictor import config

    assert (config.PROJECT_ROOT / "pyproject.toml").exists()


def test_default_paths_are_under_data_dir():
    from nba_predictor import config

    assert config.DATA_DIR == config.PROJECT_ROOT / "data"
    assert config.CACHE_DIR == config.DATA_DIR / "cache"
    assert config.TRACKING_DB_PATH == config.DATA_DIR / "tracking.db"


def test_project_root_uses_env_override_when_set(monkeypatch, tmp_path):
    import importlib

    from nba_predictor import config

    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    importlib.reload(config)

    assert config.PROJECT_ROOT == tmp_path

    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    importlib.reload(config)


def test_public_mode_defaults_false(monkeypatch):
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    import importlib

    from nba_predictor import config

    importlib.reload(config)
    assert config.PUBLIC_MODE is False


def test_public_mode_true_when_env_set(monkeypatch):
    monkeypatch.setenv("PUBLIC_MODE", "true")
    import importlib

    from nba_predictor import config

    importlib.reload(config)
    assert config.PUBLIC_MODE is True
    monkeypatch.delenv("PUBLIC_MODE", raising=False)
    importlib.reload(config)


def test_ensure_cache_dirs_creates_all_subdirs(tmp_path, monkeypatch):
    import importlib

    from nba_predictor import config

    monkeypatch.setattr(config, "CACHE_DIR", tmp_path / "cache")
    config.ensure_cache_dirs()
    for name in config.CACHE_SUBDIRS:
        assert (tmp_path / "cache" / name).is_dir()
