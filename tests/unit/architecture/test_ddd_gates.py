"""Testes arquiteturais automatizados para invariantes DDD, Hexagonal e limite de linhas."""

from __future__ import annotations

import ast
from pathlib import Path

from src.application.ports.outbound import LayaDecisionPort
from src.infrastructure.inference.laya_decision_adapter import LayaDecisionAdapter


_REPO_ROOT = Path(__file__).resolve().parents[3]
_FORBIDDEN_DOMAIN_PREFIXES = (
    "src.infrastructure",
    "src.application",
    "src.presentation",
    "laya",
    "torch",
    "redis",
    "asyncpg",
    "aiohttp",
    "websockets",
    "pandas",
)


def test_domain_has_no_forbidden_imports() -> None:
    """Garante pureza absoluta da camada de dominio (zero infraestrutura/frameworks)."""
    violations: list[str] = []
    for path in sorted((_REPO_ROOT / "src" / "domain").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.startswith(_FORBIDDEN_DOMAIN_PREFIXES):
                    violations.append(f"{path.relative_to(_REPO_ROOT)} -> {name}")
    assert not violations, f"Imports proibidos em src/domain: {violations}"


def test_src_files_respect_300_line_hard_gate() -> None:
    """Garante o hard gate de no maximo 300 linhas por modulo Python em src/."""
    oversized = [
        (str(path.relative_to(_REPO_ROOT)), len(path.read_text(encoding="utf-8").splitlines()))
        for path in sorted((_REPO_ROOT / "src").rglob("*.py"))
    ]
    offenders = [(p, n) for p, n in oversized if n > 300]
    assert not offenders, f"Modulos acima de 300 linhas: {offenders}"


def test_laya_decision_adapter_satisfies_outbound_port() -> None:
    """Garante conformidade estrutural do adapter Laya com a porta hexagonal."""
    adapter = LayaDecisionAdapter(model_id="convaiinnovations/laya", device="cpu", context_candles=6)
    assert isinstance(adapter, LayaDecisionPort)
