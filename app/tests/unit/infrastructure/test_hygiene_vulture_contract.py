"""Contrato de higiene: Vulture confidence 100 + allowlist no CI."""

from __future__ import annotations

import tomllib
from pathlib import Path


def _app_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_vulture_min_confidence_is_100():
    pyproject = _app_root() / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    conf = data["tool"]["vulture"]["min_confidence"]
    assert int(conf) == 100
    paths = data["tool"]["vulture"]["paths"]
    assert ".vulture_whitelist.py" in paths


def test_vulture_whitelist_file_exists():
    assert (_app_root() / ".vulture_whitelist.py").is_file()


def test_clean_workspace_invokes_vulture_whitelist_and_confidence():
    src = (_app_root() / "scripts" / "operations" / "clean_workspace.py").read_text(encoding="utf-8")
    assert ".vulture_whitelist.py" in src
    assert '"--min-confidence"' in src or "'--min-confidence'" in src
    assert '"100"' in src or "'100'" in src


def test_clean_workspace_locks_coverage_at_100_percent():
    src = (_app_root() / "scripts" / "operations" / "clean_workspace.py").read_text(encoding="utf-8")
    assert "if int(fail_under) != 100:" in src


def test_security_audits_only_declared_project_requirements():
    src = (_app_root() / "scripts" / "operations" / "clean_workspace.py").read_text(encoding="utf-8")
    assert 'requirements = ("-r", "requirements.txt", "-r", "requirements-dev.txt")' in src
