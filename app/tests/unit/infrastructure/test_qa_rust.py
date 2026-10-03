"""Testes unitarios para os gates de qualidade da area Rust."""

from pathlib import Path

import pytest

from scripts.operations.qa.common import AREAS
from scripts.operations.qa.dispatch import run_area_stage
from scripts.operations.qa.rust import _manifest_path, run_rust


def test_qa_rust_area_in_common_areas():
    assert "rust" in AREAS


def test_qa_rust_manifest_path():
    root = Path("/dummy/root")
    manifest = _manifest_path(root)
    assert manifest.name == "Cargo.toml"
    assert "native" in manifest.parts
    assert "aether_core" in manifest.parts


def test_qa_rust_missing_manifest_skips(tmp_path: Path):
    run_rust("lint", tmp_path)
    run_rust("validate", tmp_path)
    run_rust("security", tmp_path)
    run_rust("test", tmp_path)
    run_rust("build", tmp_path)


def test_qa_dispatch_routes_rust(tmp_path: Path):
    manifest_dir = tmp_path / "native" / "aether_core"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest_dir / "Cargo.toml"
    manifest.write_text("[package]\nname = 'test'\nversion = '0.1.0'\n")

    run_area_stage("rust", "validate", tmp_path)
    run_area_stage("rust", "test", tmp_path)
    run_area_stage("rust", "build", tmp_path)


def test_qa_dispatch_unknown_stage(tmp_path: Path):
    with pytest.raises(ValueError):
        run_area_stage("rust", "unknown_stage", tmp_path)
