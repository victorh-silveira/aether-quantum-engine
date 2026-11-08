"""Metricas rolling live (WR/Brier/ECE) por simbolo para ranking e telemetria."""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from src.application.services.live_signal_metrics_config import load_live_signal_metrics_from_settings
from src.application.services.log_dedupe import log_debug_if_changed
from src.domain.analytics.sample_size_policy import attach_sample_size_metrics, load_sample_size_policy


logger = logging.getLogger("AETH")


def _live() -> dict:
    """Carrega live_signal_metrics de settings."""
    return load_live_signal_metrics_from_settings()


def reset_live_signal_metrics(orch: Any | None = None) -> None:
    """Limpa o bag de metricas live no orquestrador."""
    if orch is not None and hasattr(orch, "_live_signal_metrics"):
        orch._live_signal_metrics = {}


def _ensure_bag(orch: Any) -> dict[str, deque]:
    """Garante e retorna o dicionario de historicos live por simbolo."""
    bag = getattr(orch, "_live_signal_metrics", None)
    if bag is None:
        orch._live_signal_metrics = {}
        bag = orch._live_signal_metrics
    return bag


def _ece(probs: list[float], labels: list[float], *, bins: int | None = None) -> float:
    """Calcula Expected Calibration Error em bins uniformes."""
    if bins is None:
        bins = int(_live()["ece_bins"])
    n = len(probs)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo = b / float(bins)
        hi = (b + 1) / float(bins)
        idxs = [i for i, p in enumerate(probs) if (lo <= p < hi) or (b == bins - 1 and p >= lo and p <= hi + 1e-12)]
        if not idxs:
            continue
        conf = sum(probs[i] for i in idxs) / float(len(idxs))
        acc = sum(labels[i] for i in idxs) / float(len(idxs))
        total += (len(idxs) / float(n)) * abs(conf - acc)
    return float(total)


def record_live_signal_outcome(
    orch: Any,
    symbol: str,
    *,
    won: bool,
    raw_prob: float | None,
    direction: str | None = None,
) -> dict[str, float | int]:
    """Registra outcome live e retorna snapshot atualizado do simbolo."""
    bag = _ensure_bag(orch)
    sym = str(symbol)
    hist = bag.get(sym)
    if hist is None:
        hist = deque(maxlen=int(_live()["window"]))
        bag[sym] = hist
    prob = 0.5 if raw_prob is None else max(0.0, min(1.0, float(raw_prob)))
    dir_name = str(direction or "").upper()
    label = (
        (1.0 if bool(won) == (dir_name == "CALL") else 0.0) if dir_name in {"CALL", "PUT"} else (1.0 if won else 0.0)
    )
    hist.append((bool(won), float(prob), float(label)))
    snap = live_signal_snapshot(orch, sym)
    if int(snap.get("live_n", 0)) >= int(_live()["min_rank"]) and int(snap.get("live_n", 0)) % 8 == 0:
        logger.debug(
            "LIVE_SIGNAL | %s | wr=%.2f | brier=%.3f | ece=%.3f | n=%d",
            sym,
            float(snap.get("live_wr", 0.0)),
            float(snap.get("live_brier", 1.0)),
            float(snap.get("live_ece", 0.0)),
            int(snap.get("live_n", 0)),
        )
    return snap


def live_signal_snapshot(orch: Any, symbol: str) -> dict[str, float | int]:
    """Retorna WR/Brier/ECE rolling do simbolo a partir do bag live."""
    bag = getattr(orch, "_live_signal_metrics", None)
    if not isinstance(bag, dict):
        return {"live_n": 0, "live_wr": 0.0, "live_brier": 1.0, "live_ece": 0.0}
    hist = bag.get(str(symbol))
    if not hist:
        return {"live_n": 0, "live_wr": 0.0, "live_brier": 1.0, "live_ece": 0.0}
    wins = [1.0 if row[0] else 0.0 for row in hist]
    probs = [float(row[1]) for row in hist]
    labels = [float(row[2]) for row in hist]
    n = len(hist)
    wr = sum(wins) / float(n)
    brier = sum((probs[i] - labels[i]) ** 2 for i in range(n)) / float(n)
    ece = _ece(probs, labels)
    return {
        "live_n": int(n),
        "live_wr": float(wr),
        "live_brier": float(brier),
        "live_ece": float(ece),
    }


def attach_live_signal_metrics(orch: Any | None, symbol: str, metrics: dict[str, Any]) -> None:
    """Injeta live_n/wr/brier/ece no dicionario de metricas de decisao."""
    if orch is None or not isinstance(metrics, dict):
        return
    snap = live_signal_snapshot(orch, symbol)
    metrics["live_n"] = int(snap["live_n"])
    metrics["live_wr"] = float(snap["live_wr"])
    metrics["live_brier"] = float(snap["live_brier"])
    metrics["live_ece"] = float(snap["live_ece"])
    attach_sample_size_metrics(metrics, int(snap["live_n"]))


def apply_live_calib_drift_soft(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    symbol: str | None = None,
) -> bool:
    """Marca soft veto de calib drift so com N suficiente (Lei dos Grandes Numeros)."""
    n = int(metrics.get("live_n", 0) or 0)
    ece = metrics.get("live_ece")
    wr = metrics.get("live_wr")
    raw = metrics.get("raw_prob")
    metrics["calib_drift_soft"] = False
    metrics["calib_drift_soft_penalty"] = 0.0
    if ece is None or wr is None or raw is None:
        return False
    policy = load_sample_size_policy()
    if n < int(policy["calib_soft_min_n"]):
        return False
    raw_side = max(float(raw), 1.0 - float(raw))
    inconsistent = abs(float(wr) - float(raw_side)) > 0.12
    if float(ece) <= float(_live()["ece_soft_threshold"]) or not inconsistent:
        return False
    metrics["calib_drift_soft"] = True
    metrics["calib_drift_soft_penalty"] = float(_live()["drift_soft_penalty"])
    metrics["calib_drift_reason"] = "CALIB_DRIFT_SOFT"
    if orch is not None:
        cycle = int(getattr(orch, "_active_cycle_id", 0) or 0)
        log_debug_if_changed(
            orch,
            logger,
            f"calib_drift_soft:{cycle}:{symbol or '?'}",
            "1",
            "CALIB_DRIFT_SOFT | ece=%.3f | live_wr=%.2f | raw_side=%.2f | n=%d",
            float(ece),
            float(wr),
            float(raw_side),
            n,
        )
    if n >= int(_live()["drift_soft_veto_n"]):
        base = metrics.get("trade_score")
        if base is None:
            base = metrics.get("resolved_conviction")
        if base is None:
            base = raw_side
        compressed = max(float(_live()["drift_min_score"]), float(base) * float(_live()["drift_score_factor"]))
        metrics["trade_score"] = compressed
        metrics["conviction"] = compressed
        metrics["signal_status"] = "SOFT_VETO"
        metrics["calib_drift_soft_veto"] = True
    return True
