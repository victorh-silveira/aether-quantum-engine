"""Testes unitarios para os gates de qualidade da area Rust."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from scripts.operations.qa.common import AREAS
from scripts.operations.qa.dispatch import run_area_stage
from scripts.operations.qa.rust import _manifest_path, _rust_env, run_rust


def test_qa_rust_area_in_common_areas():
    """Valida presenca de rust nas areas de QA."""
    assert "rust" in AREAS


def test_qa_rust_manifest_path():
    """Valida resolucao de caminho do manifesto Cargo.toml."""
    root = Path("/dummy/root")
    manifest = _manifest_path(root)
    assert manifest.name == "Cargo.toml"
    assert "native" in manifest.parts
    assert "aether_core" in manifest.parts


def test_qa_rust_env():
    """Valida configuracao de variavel para CPython Stable ABI."""
    env = _rust_env()
    assert env.get("PYO3_USE_ABI3_FORWARD_COMPATIBILITY") == "1"


def test_qa_rust_missing_manifest_skips(tmp_path: Path):
    """Garante skip gracioso quando manifesto nao existe."""
    run_rust("lint", tmp_path)
    run_rust("validate", tmp_path)
    run_rust("security", tmp_path)
    run_rust("test", tmp_path)
    run_rust("build", tmp_path)


def test_qa_rust_cargo_none_returns(tmp_path: Path):
    """Garante retorno imediato caso require_tool retorne None."""
    manifest_dir = tmp_path / "native" / "aether_core"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "Cargo.toml").write_text("[package]\nname = 'test'\n")

    with patch("scripts.operations.qa.rust.require_tool", return_value=None):
        run_rust("lint", tmp_path)


def test_qa_rust_lint_branches(tmp_path: Path):
    """Testa caminhos de lint com ferramentas presentes e ausentes."""
    manifest_dir = tmp_path / "native" / "aether_core"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "Cargo.toml").write_text("[package]\nname = 'test'\n")

    mock_run_cmd = MagicMock()
    with (
        patch("scripts.operations.qa.rust.require_tool", return_value="cargo"),
        patch("scripts.operations.qa.rust.run_cmd", mock_run_cmd),
        patch("scripts.operations.qa.rust.which", return_value=None),
    ):
        run_rust("lint", tmp_path)
    assert mock_run_cmd.call_count == 0

    with (
        patch("scripts.operations.qa.rust.require_tool", return_value="cargo"),
        patch("scripts.operations.qa.rust.run_cmd", mock_run_cmd),
        patch("scripts.operations.qa.rust.which", return_value="/bin/tool"),
    ):
        run_rust("lint", tmp_path)
    assert mock_run_cmd.call_count == 2


def test_qa_rust_security_branches(tmp_path: Path):
    """Testa caminhos de security com cargo-audit presente e ausente."""
    manifest_dir = tmp_path / "native" / "aether_core"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "Cargo.toml").write_text("[package]\nname = 'test'\n")

    mock_run_cmd = MagicMock()
    with (
        patch("scripts.operations.qa.rust.require_tool", return_value="cargo"),
        patch("scripts.operations.qa.rust.run_cmd", mock_run_cmd),
        patch("scripts.operations.qa.rust.which", return_value=None),
    ):
        run_rust("security", tmp_path)
    assert mock_run_cmd.call_count == 0

    with (
        patch("scripts.operations.qa.rust.require_tool", return_value="cargo"),
        patch("scripts.operations.qa.rust.run_cmd", mock_run_cmd),
        patch("scripts.operations.qa.rust.which", return_value="/bin/cargo-audit"),
    ):
        run_rust("security", tmp_path)
    assert mock_run_cmd.call_count == 1


def test_qa_rust_other_stages(tmp_path: Path):
    """Testa validate, test, build e estagio nao aplicavel."""
    manifest_dir = tmp_path / "native" / "aether_core"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "Cargo.toml").write_text("[package]\nname = 'test'\n")

    mock_run_cmd = MagicMock()
    with (
        patch("scripts.operations.qa.rust.require_tool", return_value="cargo"),
        patch("scripts.operations.qa.rust.run_cmd", mock_run_cmd),
    ):
        run_rust("validate", tmp_path)
        run_rust("test", tmp_path)
        run_rust("build", tmp_path)
        run_rust("desconhecido", tmp_path)

    assert mock_run_cmd.call_count == 3


def test_qa_dispatch_routes_rust(tmp_path: Path):
    """Valida roteamento pelo dispatcher mockando chamadas ao cargo."""
    manifest_dir = tmp_path / "native" / "aether_core"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / "Cargo.toml").write_text("[package]\nname = 'test'\nversion = '0.1.0'\n")

    mock_run_cmd = MagicMock()
    with (
        patch("scripts.operations.qa.rust.require_tool", return_value="cargo"),
        patch("scripts.operations.qa.rust.run_cmd", mock_run_cmd),
    ):
        run_area_stage("rust", "validate", tmp_path)
        run_area_stage("rust", "test", tmp_path)
        run_area_stage("rust", "build", tmp_path)

    assert mock_run_cmd.call_count == 3


def test_qa_dispatch_unknown_stage(tmp_path: Path):
    """Valida rejeicao de estagio inexistente."""
    with pytest.raises(ValueError):
        run_area_stage("rust", "unknown_stage", tmp_path)
