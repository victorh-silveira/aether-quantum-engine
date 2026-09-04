"""Contrato unico de decisao de gate: HARD_SKIP | SOFT_SIZE | ALLOW."""

from __future__ import annotations

from typing import Any


VERDICT_HARD_SKIP = "HARD_SKIP"
VERDICT_SOFT_SIZE = "SOFT_SIZE"
VERDICT_ALLOW = "ALLOW"

_SOFT_FLAGS = (
    "loss_clf_soft",
    "cal_margin_soft",
    "neg_edge_soft",
    "mini_pair_soft",
    "regime_chop_soft",
    "fusion_weak_ev_soft",
    "anti_loss_soft",
)


def stamp_hard_skip(metrics: dict[str, Any], reason: str) -> None:
    """Marca veto tecnico HARD; limpa soft de sizing do mesmo ciclo."""
    metrics["gate_verdict"] = VERDICT_HARD_SKIP
    metrics["gate_verdict_reason"] = str(reason or "hard_skip")
    metrics.pop("neg_edge_soft", None)
    metrics.pop("neg_edge_soft_kelly_mult", None)


def stamp_soft_size(metrics: dict[str, Any], reason: str) -> None:
    """Marca EXEC permitido com sizing comprimido; bloqueia Single-Strike."""
    if str(metrics.get("gate_verdict") or "").upper() == VERDICT_HARD_SKIP:
        return
    metrics["gate_verdict"] = VERDICT_SOFT_SIZE
    metrics["gate_verdict_reason"] = str(reason or "soft_size")


def stamp_allow(metrics: dict[str, Any], reason: str = "allow") -> None:
    """Marca ciclo liberado para Single-Strike quando nenhum soft/hard ativo."""
    if str(metrics.get("gate_verdict") or "").upper() in {VERDICT_HARD_SKIP, VERDICT_SOFT_SIZE}:
        return
    metrics["gate_verdict"] = VERDICT_ALLOW
    metrics["gate_verdict_reason"] = str(reason or "allow")


def is_soft_size(metrics: dict[str, Any] | None) -> bool:
    """True quando o ciclo esta em SOFT_SIZE ou ha flag soft de sinal."""
    if not isinstance(metrics, dict):
        return False
    if str(metrics.get("gate_verdict") or "").strip().upper() == VERDICT_SOFT_SIZE:
        return True
    return any(bool(metrics.get(flag)) for flag in _SOFT_FLAGS)


def blocks_single_strike(metrics: dict[str, Any] | None) -> bool:
    """True quando Single-Strike / stop-win boost nao deve elevar stake."""
    return is_soft_size(metrics)
