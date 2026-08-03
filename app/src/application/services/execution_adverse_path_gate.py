"""Gate de path micro: bloqueia entrada contra momentum RSI/vela formada."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_direction_discordance import _macro_indicator_float
from src.application.services.execution_quality_gate_microstructure import resolve_skipped_cycles
from src.application.services.execution_quality_gate_starvation import starvation_decay_factor
from src.application.services.execution_tcn_conviction import tcn_direction_lock_active, tcn_direction_margin
from src.domain.models.trade import TradeDirection


def _adverse_path_cfg(exec_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve knobs de adverse_path a partir do exec_cfg."""
    cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    block = cfg.get("adverse_path") if isinstance(cfg.get("adverse_path"), dict) else {}
    return {
        "enabled": bool(block.get("enabled", False)),
        "rsi_bias_min": float(block.get("rsi_bias_min", 0.08)),
        "rsi_hard_bias": float(block.get("rsi_hard_bias", 0.12)),
        "waiver_margin": float(block.get("waiver_margin", 0.40)),
    }


def _rsi_opposes_proposed(rsi: float, proposed: TradeDirection, rsi_bias_min: float) -> bool:
    """True quando o RSI tem vies claro contra a direcao proposta."""
    rsi_bias = float(rsi) - 0.5
    if abs(rsi_bias) + 1e-12 < float(rsi_bias_min):
        return False
    rsi_call = rsi_bias > 0.0
    want_call = proposed == TradeDirection.CALL
    return rsi_call != want_call


def _candle_confirms_rsi(metrics: dict[str, Any], rsi: float) -> bool | None:
    """True/False se vela confirma RSI; None se nao ha vela utilizavel."""
    candle_raw = str(metrics.get("candle_color_direction") or "").upper()
    if candle_raw not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return None
    candle_call = candle_raw == TradeDirection.CALL.name
    rsi_call = float(rsi) > 0.5
    return candle_call == rsi_call


def apply_adverse_micro_path_gate(
    metrics: dict[str, Any],
    proposed: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> bool:
    """True quando o path micro opoe a direcao proposta e o ciclo deve ser skipped."""
    thr = _adverse_path_cfg(exec_cfg)
    if not bool(thr["enabled"]):
        return False
    if not tcn_direction_lock_active(metrics):
        metrics["adverse_micro_path_weak_tcn_defer"] = True
        return False
    rsi = _macro_indicator_float(metrics, "rsi")
    if rsi is None:
        return False
    if not _rsi_opposes_proposed(float(rsi), proposed, float(thr["rsi_bias_min"])):
        return False
    candle_ok = _candle_confirms_rsi(metrics, float(rsi))
    if candle_ok is False:
        return False
    hard = abs(float(rsi) - 0.5) + 1e-12 >= float(thr["rsi_hard_bias"])
    metrics["adverse_micro_path"] = True
    metrics["adverse_micro_path_rsi"] = float(rsi)
    if candle_ok is True:
        metrics["adverse_micro_path_candle"] = str(metrics.get("candle_color_direction"))
    if not hard and tcn_direction_margin(metrics) + 1e-12 >= float(thr["waiver_margin"]):
        metrics["adverse_micro_path_margin_waiver"] = True
        metrics.pop("adverse_micro_path", None)
        return False
    if hard:
        metrics["adverse_micro_path_hard"] = True
    skipped = resolve_skipped_cycles(skipped_cycles_counter=skipped_cycles_counter, orch=orch)
    decay = starvation_decay_factor(skipped, exec_cfg=exec_cfg if isinstance(exec_cfg, dict) else None)
    if decay + 1e-12 < 1.0 and not hard:
        metrics["adverse_micro_path_starvation_waiver"] = True
        metrics.pop("adverse_micro_path", None)
        return False
    metrics["gate_reason"] = "adverse_micro_path"
    metrics["quality_guard_reject"] = True
    metrics["regime_skip_cycle"] = True
    return True
