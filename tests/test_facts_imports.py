"""/facts must not import a module that does not exist.

`_player_rows` did `from ..services.hub_cache import load_player_name_map`. The
function lives in `hub_service`; there is no `hub_cache` module. The import sits
inside the function, so it escaped collection and every other test — it only
fires when a bundle actually reaches the players block. In production that is
the common case, so /facts 500'd for any game with player rows.

This asserts the real thing: every `from ..services.X import` in facts.py must
resolve, checked by importing rather than by reading.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import pytest

FACTS = Path(__file__).resolve().parents[1] / "src" / "nba_predictor" / "api" / "facts.py"


def _service_imports() -> list[str]:
    """Module names facts.py imports from nba_predictor.services, by AST.

    Handles both spellings: absolute (`nba_predictor.services.x`) and the
    relative form facts.py actually uses (`from ..services.x`, level 2 from
    inside nba_predictor.api).
    """
    tree = ast.parse(FACTS.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.level:
            # nba_predictor.api.facts -> level 2 is nba_predictor
            prefix = ".".join(node.module.split(".")[:1])
            if prefix != "services":
                continue
            names.append(f"nba_predictor.{node.module}")
        elif node.module.startswith("nba_predictor.services"):
            names.append(node.module)
    return names


def test_facts_imports_at_least_one_services_module():
    """Guards the sweep below against passing because it found nothing."""
    assert _service_imports(), "expected facts.py to import from nba_predictor.services"


def test_every_services_module_facts_imports_exists():
    missing = []
    for module in _service_imports():
        try:
            importlib.import_module(module)
        except ModuleNotFoundError:
            missing.append(module)
    assert not missing, f"facts.py imports modules that do not exist: {missing}"


def test_load_player_name_map_lives_where_facts_looks_for_it():
    """The specific break, pinned directly: the name facts.py uses must be
    importable from the module it names."""
    facts = importlib.import_module("nba_predictor.api.facts")
    assert hasattr(facts, "_player_rows")
    from nba_predictor.services.hub_service import load_player_name_map

    assert callable(load_player_name_map)


def test_every_services_module_in_the_package_imports():
    """A cheap sweep, so the next missing module is caught here rather than
    by a 500 in production."""
    import nba_predictor.services as services

    failures = []
    for info in pkgutil.iter_modules(services.__path__):
        try:
            importlib.import_module(f"nba_predictor.services.{info.name}")
        except Exception as exc:  # noqa: BLE001 - the point is to report anything
            failures.append(f"{info.name}: {type(exc).__name__}: {exc}")
    assert not failures, "services modules failed to import: " + "; ".join(failures)
