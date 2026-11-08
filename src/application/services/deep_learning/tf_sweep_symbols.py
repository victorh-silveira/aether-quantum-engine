"""Universo de simbolos do sweep e escrita do SSOT drift_symbols."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aether_paths import REPO_ROOT


DEFAULT_SWEEP_SYMBOLS: tuple[str, ...] = ("1HZ75V",)

_DRIFT_TEMPLATE = '''"""Simbolos de trading Deriv (universo single-symbol {symbol})."""

TRADING_SYMBOLS: tuple[str, ...] = ("{symbol}",)
DRIFT_SYMBOLS: tuple[str, ...] = TRADING_SYMBOLS
DEFAULT_ANCHOR = "{symbol}"

HEDGE_PEER: dict[str, str] = {{}}

HIGH_SIDE: frozenset[str] = frozenset()
LOW_SIDE: frozenset[str] = frozenset()

_SYMBOL_ORDER = {{symbol: index for index, symbol in enumerate(TRADING_SYMBOLS)}}


def hedge_peer(symbol: str) -> str | None:
    """Retorna o simbolo par de hedge ou None quando nao ha par configurado."""
    return HEDGE_PEER.get(str(symbol))


def is_high_side(symbol: str) -> bool:
    """True quando o simbolo esta no lado high do universo (vazio em single-symbol)."""
    return str(symbol) in HIGH_SIDE


def sym_is_low_barrier(symbol: str, peer: str | None = None) -> bool:
    """True quando o simbolo e low-barrier relativo ao peer (sempre False sem peer)."""
    peer_key = peer if peer is not None else hedge_peer(symbol)
    if peer_key is None:
        return False
    return _SYMBOL_ORDER.get(str(symbol), 0) < _SYMBOL_ORDER.get(str(peer_key), 0)
'''


def normalize_sweep_symbol(raw: Any) -> str:
    """Normaliza ticker Deriv do cluster de volatilidade."""
    return str(raw).strip().upper()


def resolve_sweep_symbols(knobs: dict[str, Any] | None = None) -> list[str]:
    """Lista de simbolos do sweep (fail-closed ao default do cluster)."""
    block = knobs if isinstance(knobs, dict) else {}
    raw = block.get("symbols")
    if not isinstance(raw, list) or not raw:
        return list(DEFAULT_SWEEP_SYMBOLS)
    out: list[str] = []
    for item in raw:
        sym = normalize_sweep_symbol(item)
        if sym and sym not in out:
            out.append(sym)
    return out or list(DEFAULT_SWEEP_SYMBOLS)


def patch_settings_for_symbol(settings: dict[str, Any], symbol: str) -> dict[str, Any]:
    """Fixa anchor/symbols/train_symbols no ticker promovido ou em treino isolado."""
    sym = normalize_sweep_symbol(symbol)
    settings["anchor"] = sym
    settings["symbols"] = [sym]
    dl = settings.setdefault("deep_learning", {})
    if isinstance(dl, dict):
        dl["train_symbols"] = [sym]
    return settings


def write_trading_symbols_module(symbol: str, *, path: Path | None = None) -> Path:
    """Reescreve drift_symbols.py com universo single-symbol do vencedor."""
    sym = normalize_sweep_symbol(symbol)
    target = path if path is not None else REPO_ROOT / "app" / "src" / "domain" / "symbols" / "drift_symbols.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_DRIFT_TEMPLATE.format(symbol=sym), encoding="utf-8")
    return target


def clear_other_live_checkpoints(dest_dir: Path, symbol: str) -> list[Path]:
    """Remove .pth/.pt de outros simbolos em data/dl apos promote."""
    sym = normalize_sweep_symbol(symbol)
    removed: list[Path] = []
    if not dest_dir.is_dir():
        return removed
    keep = {f"{sym}.pth", f"{sym}_ts.pt"}
    for path in dest_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if (name.endswith(".pth") or name.endswith("_ts.pt")) and name not in keep:
            path.unlink(missing_ok=True)
            removed.append(path)
    return removed
