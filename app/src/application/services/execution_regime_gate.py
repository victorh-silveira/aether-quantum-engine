"""Gate boolean de regime: ADX fraco + squeeze BB -> HARD SKIP sem alterar lado."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_gate_verdict import stamp_hard_skip
from src.application.services.regime_micro_freeze import severe_bb_compression
from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_keys


logger = logging.getLogger("AETH")


def parse_regime_gate_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve knobs regime_gate em orchestrator.execution.signal_skip."""
    block = merge_settings_block(("orchestrator", "execution", "signal_skip"), raw)
    require_keys(
        block,
        ("regime_gate_enabled", "regime_adx_max", "regime_bb_squeeze_enabled"),
        "orchestrator.execution.signal_skip",
    )
    adx_max = require_float(block, "regime_adx_max")
    if adx_max <= 0.0 or adx_max > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.regime_adx_max deve estar em (0, 1]")
    return {
        "regime_gate_enabled": require_bool(block, "regime_gate_enabled"),
        "regime_adx_max": adx_max,
        "regime_bb_squeeze_enabled": require_bool(block, "regime_bb_squeeze_enabled"),
    }


def _indicator_float(metrics: dict[str, Any], key: str) -> float | None:
    """Le indicador de metrics.indicators com fallback micro_indicators."""
    for bucket_name in ("indicators", "micro_indicators"):
        bucket = metrics.get(bucket_name)
        if not isinstance(bucket, dict) or key not in bucket:
            continue
        try:
            return float(bucket[key])
        except (TypeError, ValueError):
            return None
    return None


def apply_regime_boolean_gate(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """HARD SKIP se regime nao operable (ADX baixo + BB squeeze). Nao altera CALL/PUT."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status == "SKIP" or status.startswith("SKIP:"):
        return False
    vision = cfg if isinstance(cfg, dict) and "regime_gate_enabled" in cfg else parse_regime_gate_config(cfg)
    if not bool(vision.get("regime_gate_enabled", False)):
        return False
    adx = _indicator_float(metrics, "adx")
    if adx is None:
        return False
    adx_max = float(vision["regime_adx_max"])
    adx_weak = adx + 1e-12 < adx_max
    metrics["regime_gate_adx"] = float(adx)
    metrics["regime_gate_adx_weak"] = bool(adx_weak)
    squeeze = False
    if bool(vision.get("regime_bb_squeeze_enabled", True)):
        squeeze = bool(severe_bb_compression(metrics))
    metrics["regime_gate_bb_squeeze"] = bool(squeeze)
    if not (adx_weak and squeeze):
        return False
    reason = "regime_squeeze"
    metrics["execution_candidate_ready"] = False
    metrics["gate_reason"] = reason
    metrics["signal_status"] = "SKIP:REGIME_SQUEEZE"
    stamp_hard_skip(metrics, reason)
    if orch is not None:
        logger.info(
            "REGIME || HARD_SKIP why=%s adx=%.4f thr=%.4f squeeze=%s",
            reason,
            float(adx),
            adx_max,
            "yes" if squeeze else "no",
        )
    return True
