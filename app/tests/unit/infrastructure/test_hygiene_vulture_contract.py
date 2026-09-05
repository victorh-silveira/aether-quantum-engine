"""Contrato de higiene: Vulture confidence 80 + allowlist no CI."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _app_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_vulture_min_confidence_is_80():
    pyproject = _app_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    conf = data["tool"]["vulture"]["min_confidence"]
    assert int(conf) == 80
    paths = data["tool"]["vulture"]["paths"]
    assert ".vulture_whitelist.py" in paths


def test_vulture_whitelist_file_exists():
    assert (_app_root() / ".vulture_whitelist.py").is_file()


def test_clean_workspace_invokes_vulture_whitelist_and_confidence():
    src = (_app_root() / "scripts" / "operations" / "clean_workspace.py").read_text(encoding="utf-8")
    assert ".vulture_whitelist.py" in src
    assert '"--min-confidence"' in src or "'--min-confidence'" in src
    assert '"80"' in src or "'80'" in src
