"""Garantia de domain puro: sem I/O libs nem frameworks de ML/async."""

from __future__ import annotations

import ast
from pathlib import Path

from aether_paths import repo_path


_BANNED = frozenset(
    {
        "asyncio",
        "torch",
        "polars",
        "httpx",
        "redis",
        "asyncpg",
        "websockets",
        "fastapi",
        "uvicorn",
        "pandas",
        "minio",
        "joblib",
        "lightgbm",
        "optuna",
        "sklearn",
    }
)


def _iter_domain_py() -> list[Path]:
    root = repo_path("app", "src", "domain")
    return sorted(root.rglob("*.py"))


def _imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_domain_has_no_banned_imports():
    offenders: list[str] = []
    for path in _iter_domain_py():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        banned = _imported_roots(tree) & _BANNED
        if banned:
            offenders.append(f"{path.relative_to(repo_path())}: {sorted(banned)}")
    assert not offenders, "domain importou libs proibidas:\n" + "\n".join(offenders)


def test_trade_and_candle_use_slots():
    from src.domain.models.market_data import Candle
    from src.domain.models.trade import Contract, Proposal, TradeResult

    assert hasattr(Proposal, "__slots__")
    assert hasattr(Contract, "__slots__")
    assert hasattr(TradeResult, "__slots__")
    assert hasattr(Candle, "__slots__")
