"""Soft Kelly em regime chop (ADX fraco + Hurst random-walk ou SCALE chop)."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_signal_skip import apply_kelly_soft
from src.application.services.log_dedupe import log_info_if_changed
from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_keys


logger = logging.getLogger("AETH")


def parse_regime_chop_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve knobs chop em orchestrator.execution.signal_skip."""
    block = merge_settings_block(("orchestrator", "execution", "signal_skip"), raw)
    require_keys(
        block,
        (
            "chop_pause_enabled",
            "chop_adx_max",
            "chop_hurst_min",
            "chop_hurst_max",
            "chop_soft_kelly_mult",
        ),
        "orchestrator.execution.signal_skip",
    )
    hurst_min = require_float(block, "chop_hurst_min")
    hurst_max = require_float(block, "chop_hurst_max")
    if hurst_max < hurst_min:
        raise ValueError("orchestrator.execution.signal_skip.chop_hurst_max deve ser >= chop_hurst_min")
    soft_mult = require_float(block, "chop_soft_kelly_mult")
    if soft_mult <= 0.0 or soft_mult > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.chop_soft_kelly_mult deve estar em (0, 1]")
    return {
        "chop_pause_enabled": require_bool(block, "chop_pause_enabled"),
        "chop_adx_max": require_float(block, "chop_adx_max"),
        "chop_hurst_min": hurst_min,
        "chop_hurst_max": hurst_max,
        "chop_soft_kelly_mult": soft_mult,
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


def apply_regime_chop_pause(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Soft Kelly quando ADX fraco e Hurst em banda ou SCALE chop. True se atenuou."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status == "SKIP" or status.startswith("SKIP:"):
        return False
    vision = cfg if isinstance(cfg, dict) and "chop_pause_enabled" in cfg else parse_regime_chop_config(cfg)
    if not bool(vision.get("chop_pause_enabled", True)):
        return False
    adx = _indicator_float(metrics, "adx")
    hurst = _indicator_float(metrics, "hurst")
    if adx is None or hurst is None:
        return False
    adx_max = float(vision["chop_adx_max"])
    h_min = float(vision["chop_hurst_min"])
    h_max = float(vision["chop_hurst_max"])
    hurst_band = h_min - 1e-12 <= hurst <= h_max + 1e-12
    scale_chop = str(metrics.get("scale_micro_regime") or "").strip().lower() == "chop"
    adx_weak = adx + 1e-12 < adx_max
    if not adx_weak:
        return False
    if not (hurst_band or scale_chop):
        return False
    soft_mult = float(vision.get("chop_soft_kelly_mult", 0.55))
    apply_kelly_soft(metrics, soft_mult, waived="regime_chop_soft", flag="regime_chop_soft")
    metrics["regime_chop_adx"] = float(adx)
    metrics["regime_chop_hurst"] = float(hurst)
    metrics["regime_chop_via_scale"] = bool(scale_chop and not hurst_band)
    metrics.pop("regime_chop_pause", None)
    if orch is not None:
        log_info_if_changed(
            orch,
            logger,
            "regime_chop_soft",
            f"{adx:.4f}:{hurst:.4f}:{int(scale_chop)}:{soft_mult:.2f}",
            "REGIME || CHOP_SOFT adx=%.4f hurst=%.4f scale_chop=%s kelly_mult=%.2f",
            float(adx),
            float(hurst),
            "yes" if scale_chop else "no",
            soft_mult,
        )
    return True
