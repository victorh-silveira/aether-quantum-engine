"""Confluencia macro transatlantica entre clusters de indices US e EU."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.macro_cluster_align import (
    cluster_trade_direction,
    expected_cluster_tags_line,
    reconcile_cluster_tags_with_macro,
)
from src.application.services.llm.macro_config import ClusterVote, MacroSnapshot, resolve_macro_config
from src.application.services.llm.macro_fx_reference import fx_reference_context_line


__all__ = [
    "ClusterVote",
    "MacroSnapshot",
    "aggregate_cluster_vote",
    "apply_m5_fallback_to_snapshot",
    "build_macro_snapshot",
    "classify_transatlantic_confluence",
    "cluster_direction_from_closes",
    "cluster_trade_direction",
    "empty_macro_snapshot",
    "eurusd_bias_from_confluence",
    "expected_cluster_tags_line",
    "format_macro_confluence_block",
    "fx_reference_context_line",
    "macro_snapshot_from_votes",
    "reconcile_cluster_tags_with_macro",
    "resolve_macro_config",
]


def _display_move_token(internal_dir: str) -> str:
    """Converte direcao interna up/down/flat para RISE/FALL/FLAT no prompt."""
    if internal_dir == "up":
        return "RISE"
    if internal_dir == "down":
        return "FALL"
    return "FLAT"


def cluster_direction_from_closes(
    closes: list[float],
    threshold_pct: float,
    min_move_pct: float | None = None,
) -> str:
    """Classifica direcao de um indice a partir de fechamentos recentes."""
    if len(closes) < 2:
        return "flat"
    start = float(closes[0])
    end = float(closes[-1])
    if start == 0:
        return "flat"
    ret = ((end - start) / start) * 100.0
    floor = float(min_move_pct) if min_move_pct is not None else float(threshold_pct)
    if abs(ret) < floor:
        return "flat"
    if ret > threshold_pct:
        return "up"
    if ret < -threshold_pct:
        return "down"
    return "flat"


def aggregate_cluster_vote(
    symbols: list[str],
    closes_map: dict[str, list[float]],
    *,
    threshold_pct: float,
    min_indices: int,
    labels: dict[str, Any] | None = None,
    region: str = "us",
    min_move_pct: float | None = None,
) -> ClusterVote:
    """Agrega votos up/down/flat de um cluster e retorna direcao majoritaria."""
    parts: list[str] = []
    votes: dict[str, int] = {"up": 0, "down": 0, "flat": 0}
    label_cfg = labels if isinstance(labels, dict) else {}
    cluster_labels = label_cfg.get("cluster_labels") if isinstance(label_cfg.get("cluster_labels"), dict) else {}
    display_labels = cluster_labels.get(region, []) if isinstance(cluster_labels, dict) else []

    for i, sym in enumerate(symbols):
        closes = closes_map.get(sym, [])
        direction = cluster_direction_from_closes(closes, threshold_pct, min_move_pct)
        votes[direction] = votes.get(direction, 0) + 1
        if isinstance(closes, list) and len(closes) >= 2 and float(closes[0]) != 0:
            last_px = float(closes[-1])
            ret = ((closes[-1] - closes[0]) / closes[0]) * 100.0
            name = (display_labels[i] if i < len(display_labels) else sym.replace("OTC_", "")).upper()
            move = _display_move_token(direction)
            parts.append(f"{name}: {last_px:.2f} ({move} {ret:+.2f}%)")
        else:
            name = (display_labels[i] if i < len(display_labels) else sym.replace("OTC_", "")).upper()
            parts.append(f"{name}: N/A")

    total = len(symbols)
    if total < min_indices:
        return ClusterVote("flat", 0.0, tuple(parts))

    best_dir = max(votes, key=lambda k: votes[k])
    if votes[best_dir] < min_indices:
        return ClusterVote("flat", 0.0, tuple(parts))

    strength = float(votes[best_dir]) / float(total) if total else 0.0
    return ClusterVote(best_dir, strength, tuple(parts))


def classify_transatlantic_confluence(us_dir: str, eu_dir: str) -> str:
    """Classifica confluencia entre clusters US e EU."""
    if us_dir == "flat" or eu_dir == "flat":
        return "indefinido"
    if us_dir == "up" and eu_dir == "up":
        return "risk_on"
    if us_dir == "down" and eu_dir == "down":
        return "risk_off"
    if us_dir == "up" and eu_dir == "down":
        return "divergence_us_leads"
    if us_dir == "down" and eu_dir == "up":
        return "divergence_eu_leads"
    return "indefinido"


def eurusd_bias_from_confluence(tag: str, *, us_dir: str = "flat", eu_dir: str = "flat") -> str:
    """Traduz tag de confluencia em viés sugerido para EURUSD (somente CALL ou PUT)."""
    if tag == "risk_on":
        return "CALL"
    if tag == "risk_off":
        return "PUT"
    if tag == "divergence_us_leads":
        return "CALL"
    if tag == "divergence_eu_leads":
        return "CALL"
    if us_dir == "up" or eu_dir == "up":
        return "CALL"
    return "PUT"


def format_macro_confluence_block(
    us_summary: str,
    eu_summary: str,
    tag: str,
    fx_ref: str,
    *,
    eurusd_bias: str = "",
    cluster_quant_line: str = "",
) -> str:
    """Formata bloco MACRO_CONFLUENCIA para prompt e telemetria."""
    bias = eurusd_bias or eurusd_bias_from_confluence(tag)
    base = f"MACRO_CONFLUENCIA: tag={tag} | EURUSD_bias_quant={bias} | {us_summary} | {eu_summary} | {fx_ref}"
    extra = (cluster_quant_line or "").strip()
    return f"{base} | {extra}" if extra else base


def empty_macro_snapshot() -> MacroSnapshot:
    """Retorna snapshot vazio quando dados de cluster nao estao disponiveis."""
    fx_ref = fx_reference_context_line("indefinido", {})
    return MacroSnapshot(
        us_dir="flat",
        eu_dir="flat",
        us_strength=0.0,
        eu_strength=0.0,
        tag="indefinido",
        eurusd_bias="PUT",
        cluster_status="",
        macro_block="",
        fx_reference_line=fx_ref,
        us_parts=(),
        eu_parts=(),
    )


from src.application.services.llm.macro_snapshot_build import (  # noqa: E402
    apply_m5_fallback_to_snapshot,
    build_macro_snapshot,
    macro_snapshot_from_votes,
)
