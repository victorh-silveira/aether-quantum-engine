"""Predicao e gating de execucao por simbolo no bridge Deep Learning."""

import logging
from typing import Any

from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry
from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_gating import (
    gating_block_reason,
    resolve_edge,
    resolve_gating_thresholds,
    strong_signal_bypasses_val_acc,
)
from src.application.services.deep_learning.dl_outcomes import (
    blended_val_accuracy,
    live_win_rate,
)
from src.application.services.deep_learning.dl_regime import direction_aligns_with_regime
from src.application.services.deep_learning.model import predict_next_direction


logger = logging.getLogger("AETH")


def predict_symbol_decision(
    orch,
    symbol: str,
    model,
    prices,
    norm_stats,
    runtime: dict,
    params: dict[str, Any],
    train_loss: float | None,
    *,
    recovery_active: bool,
    granularity: int = 300,
    pair_prices=None,
) -> dict:
    """Gera predicao e gating de execucao com score calibrado unificado."""
    val_accuracy = blended_val_accuracy(
        orch,
        symbol,
        float(runtime["val_accuracy"]),
        live_weight=float(params.get("val_acc_live_blend", 0.55)),
    )
    calibrator = runtime.get("calibrator") or CalibratorState()
    min_conviction, min_edge_margin, min_val_accuracy = resolve_gating_thresholds(
        params, recovery_active=recovery_active
    )
    try:
        gap = float(params.get("max_calibrated_raw_gap", 0.18))
        min_raw = float(params.get("min_raw_conviction_execute", 0.52))
        if recovery_active:
            min_raw = max(min_raw, float(params.get("recovery_min_raw_conviction", 0.58)))
        deploy_ok = bool(runtime.get("deploy_ok", False))
        gran = int(granularity or params.get("granularity", 300))
        direction, prob, trade_score, raw_prob = predict_next_direction(
            model,
            prices,
            lookback=int(runtime.get("lookback", params["lookback"])),
            norm_stats=norm_stats,
            val_accuracy=val_accuracy,
            calibrator=calibrator,
            max_calibrated_raw_gap=gap,
            min_direction_margin=float(params.get("min_direction_margin", 0.10)),
            granularity=gran,
            pair_prices=pair_prices,
            deploy_ok=deploy_ok,
        )
        if direction is None:
            entry = build_decision_entry(
                None,
                0.0,
                execute=False,
                val_accuracy=val_accuracy,
                edge=0.0,
                train_loss=train_loss,
                raw_prob=raw_prob,
            )
            entry["metrics"]["gate_reason"] = "direction_margin"
            return entry
        edge = resolve_edge(trade_score)
        raw_side = max(float(raw_prob), 1.0 - float(raw_prob))
        allow_bypass = bool(params.get("recovery_allow_bypass", False)) or not recovery_active
        block = gating_block_reason(
            trade_score,
            val_accuracy,
            min_conviction,
            min_edge_margin,
            min_val_accuracy,
            raw_prob=raw_prob,
            max_calib_gap=float(params.get("max_calib_gap_execute", gap)),
            min_raw_conviction=min_raw,
            max_val_brier=float(params.get("max_val_brier_execute", 0.26)),
            val_brier=float(runtime.get("val_brier", 1.0)),
            max_raw_saturation=float(params.get("max_raw_saturation", 0.97)),
            saturation_min_trade_score=float(params.get("saturation_min_trade_score", 0.58)),
            high_val_acc_relax=float(params.get("high_val_acc_relax", 0.68)),
            relaxed_conviction=float(params.get("relaxed_conviction", 0.54)),
            brier_untrained_floor=float(params.get("brier_untrained_floor", 0.99)),
            bypass_min_conviction=params.get("bypass_min_conviction"),
            bypass_min_edge=params.get("bypass_min_edge"),
            bypass_min_val_accuracy=float(params.get("bypass_min_val_accuracy", 0.48)),
            moderate_min_conviction=params.get("moderate_min_conviction"),
            moderate_min_edge=params.get("moderate_min_edge"),
            moderate_min_val_accuracy=params.get("moderate_min_val_accuracy"),
            allow_bypass=allow_bypass,
            deploy_ok=deploy_ok,
            recovery_active=recovery_active,
            min_conviction_for_raw_bypass=min_conviction,
        )
        execute = block is None
        regime_required = bool(params.get("require_regime_alignment", True))
        regime_ok = direction_aligns_with_regime(
            direction,
            prices,
            min_strength=float(params.get("min_regime_strength", 0.0)),
        )
        if execute and regime_required and not regime_ok:
            execute = False
            block = "regime"
        live_wr = live_win_rate(orch, symbol)
        if execute and live_wr is not None and live_wr + 1e-9 < float(params.get("min_live_win_rate", 0.42)):
            execute = False
            block = "live_wr"
        bypass_used = False
        if execute and val_accuracy + 1e-9 < min_val_accuracy:
            strong = (
                params.get("bypass_min_conviction") is not None
                and params.get("bypass_min_edge") is not None
                and allow_bypass
                and strong_signal_bypasses_val_acc(
                    raw_side,
                    edge,
                    bypass_min_conviction=params["bypass_min_conviction"],
                    bypass_min_edge=params["bypass_min_edge"],
                )
            )
            moderate = (
                params.get("moderate_min_conviction") is not None
                and params.get("moderate_min_edge") is not None
                and allow_bypass
                and strong_signal_bypasses_val_acc(
                    raw_side,
                    edge,
                    bypass_min_conviction=params["moderate_min_conviction"],
                    bypass_min_edge=params["moderate_min_edge"],
                )
            )
            if strong or moderate:
                bypass_used = True
        entry = build_decision_entry(
            direction,
            prob,
            execute=execute,
            val_accuracy=val_accuracy,
            edge=edge,
            train_loss=train_loss,
            trade_score=trade_score,
            raw_prob=raw_prob,
            val_brier=float(runtime.get("val_brier", 1.0)),
            val_ece=float(runtime.get("val_ece", 1.0)),
        )
        entry["metrics"]["gate_reason"] = block
        entry["metrics"]["bypass_val_acc"] = bypass_used
        entry["metrics"]["val_accuracy"] = val_accuracy
        if live_wr is not None:
            entry["metrics"]["live_win_rate"] = float(live_wr)
        return entry
    except Exception as e:
        logger.error("DL: Falha na predicao para %s: %s", symbol, e)
        entry = build_decision_entry(
            None,
            0.0,
            execute=False,
            val_accuracy=val_accuracy,
            edge=0.0,
            train_loss=train_loss,
        )
        entry["metrics"]["gate_reason"] = "predict_error"
        return entry
