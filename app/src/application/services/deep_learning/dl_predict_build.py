"""Montagem compartilhada de entrada de decisao DL."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry
from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    calibrate_trade_score,
    clamp_calibrated_call_to_raw_band,
)
from src.application.services.deep_learning.dl_calibration_tolerance import (
    apply_calibration_neutral_tolerance,
)
from src.application.services.deep_learning.dl_congestion import (
    series_last as _series_last,
    squeeze_congestion_active,
)
from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.deep_learning.dl_feature_matrix import build_feature_row
from src.application.services.deep_learning.dl_gating import (
    resolve_calibrated_edge,
    resolve_confidence_thresholds,
)
from src.application.services.deep_learning.dl_indicator_config import load_indicator_config_from_settings
from src.application.services.deep_learning.dl_params_blocks import parse_dynamic_threshold_config
from src.application.services.deep_learning.dl_predict_metrics import (
    attach_dynamic_metrics,
    indicators_from_series,
)
from src.application.services.deep_learning.dl_predict_telemetry import (
    prepare_meta_classifier_cross_symbol_bundle,
    stamp_micro_frame_telemetry,
)
from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model
from src.application.services.deep_learning.dl_trend import calculate_trend_direction
from src.application.services.deep_learning.model import predict_next_direction
from src.application.services.execution_sniper_gates import resolve_calibration_neutral_band
from src.application.services.execution_volatility_threshold import resolve_dynamic_threshold_bundle
from src.domain.models.trade import TradeDirection


def _series_tail(series_key: str, series: dict) -> list[float]:
    """Extrai historico numerico de uma serie de indicadores."""
    chunk = series.get(series_key)
    if chunk is None or len(chunk) == 0:
        return []
    return [float(v) for v in chunk]


def build_prediction_context(
    orch: Any,
    symbol: str,
    prices,
    runtime: dict,
    params: dict[str, Any],
    *,
    granularity: int = 60,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> dict[str, Any]:
    """Prepara series, thresholds dinamicos e calibrador para predicao DL."""
    gran = int(granularity or params.get("granularity", 3600))
    calibrator = runtime.get("calibrator") or CalibratorState()
    runtime["calibrator"] = calibrator
    series = precompute_price_series(
        prices,
        granularity=gran,
        symbol=str(symbol),
        open_=open_,
        high=high,
        low=low,
        micro=micro,
        implied_vol_bars=int(params.get("implied_vol_bars", 60)),
    )
    exec_cfg = orch.config.get("orchestrator", {}).get("execution", {}) if hasattr(orch, "config") else {}
    dynamic_cfg = parse_dynamic_threshold_config(exec_cfg if isinstance(exec_cfg, dict) else {})
    base_call, base_put = resolve_confidence_thresholds(params)
    base_edge = float(params.get("min_edge_execute", 0.04))
    bb_width = _series_last(series, "bb_width")
    atr_norm = _series_last(series, "atr_norm")
    adx = _series_last(series, "adx")
    vol_ratio = _series_last(series, "vol_ratio_short_long")
    implied_vol_ratio = _series_last(series, "implied_vol_ratio", 1.0)
    dynamic = resolve_dynamic_threshold_bundle(
        base_call=base_call,
        base_put=base_put,
        base_edge=base_edge,
        bb_width=bb_width,
        atr_norm=atr_norm,
        adx=adx,
        vol_ratio=vol_ratio,
        bb_width_history=_series_tail("bb_width", series),
        atr_norm_history=_series_tail("atr_norm", series),
        symbol=str(symbol),
        implied_vol_ratio=implied_vol_ratio,
        cfg={
            **dynamic_cfg,
            "call_base": dynamic_cfg.get("call_base", base_call),
            "put_base": dynamic_cfg.get("put_base", base_put),
            "min_edge_base": dynamic_cfg.get("min_edge_base", base_edge),
        },
    )
    call_threshold = dynamic.call_threshold if dynamic is not None else base_call
    put_threshold = dynamic.put_threshold if dynamic is not None else base_put
    return {
        "gran": gran,
        "calibrator": calibrator,
        "series": series,
        "exec_cfg": exec_cfg if isinstance(exec_cfg, dict) else {},
        "dynamic_cfg": dynamic_cfg,
        "dynamic": dynamic,
        "call_threshold": call_threshold,
        "put_threshold": put_threshold,
    }


def eager_local_predict(
    ctx: dict[str, Any],
    *,
    model,
    prices,
    norm_stats,
    runtime: dict,
    params: dict[str, Any],
    symbol: str,
    open_=None,
    high=None,
    low=None,
    micro=None,
) -> tuple[TradeDirection | None, float, float]:
    """Executa inferencia local eager com lock do modelo."""
    with guard_symbol_model(runtime):
        return predict_next_direction(
            model,
            prices,
            lookback=int(runtime.get("lookback", params["lookback"])),
            norm_stats=norm_stats,
            granularity=int(ctx["gran"]),
            symbol=str(symbol),
            open_=open_,
            high=high,
            low=low,
            micro=micro,
            implied_vol_bars=int(params.get("implied_vol_bars", 60)),
            call_threshold=float(ctx["call_threshold"]),
            put_threshold=float(ctx["put_threshold"]),
            calibrator=ctx["calibrator"],
        )


def build_prediction_entry(
    _orch: Any,
    symbol: str,
    prices,
    series: dict,
    runtime: dict,
    params: dict[str, Any],
    train_loss: float | None,
    *,
    direction: TradeDirection | None,
    prob: float,
    raw_prob: float,
    dynamic,
    dynamic_cfg: dict,
    call_threshold: float,
    put_threshold: float,
    exec_cfg: dict,
    val_accuracy: float,
) -> dict:
    """Monta dict de decisao a partir de probabilidades e series de indicadores."""
    bb_width = _series_last(series, "bb_width")
    vol_ratio = _series_last(series, "vol_ratio_short_long")
    implied_vol_ratio = _series_last(series, "implied_vol_ratio", 1.0)
    trend_dir, trend_type, trend_period, call_votes, put_votes = calculate_trend_direction(prices, series, exec_cfg)
    indicators_data = indicators_from_series(series)
    pivot = (call_threshold + put_threshold) * 0.5
    neutral_lo, neutral_hi = resolve_calibration_neutral_band(
        params.get("calibration") if isinstance(params.get("calibration"), dict) else None
    )
    calibrated_prob, resolved_dir, calibration_mode = apply_calibration_neutral_tolerance(
        float(prob),
        float(raw_prob),
        direction,
        pivot=pivot,
        neutral_lo=neutral_lo,
        neutral_hi=neutral_hi,
    )
    raw_prob = float(raw_prob)
    cal_cfg = params.get("calibration") if isinstance(params.get("calibration"), dict) else {}
    max_gap = float(cal_cfg.get("max_calibrated_raw_gap", 0.08))
    calibrated_prob, cal_capped, cal_raw_gap = clamp_calibrated_call_to_raw_band(raw_prob, calibrated_prob, max_gap)
    horizon_bars = max(1, int(params.get("label_horizon_bars", 1)))
    calibrated_edge = resolve_calibrated_edge(calibrated_prob, raw_prob=raw_prob, horizon_bars=horizon_bars)
    calibrator = runtime.get("calibrator")
    side_score = calibrate_trade_score(
        raw_prob,
        val_accuracy,
        calibrator,
        max_calibrated_raw_gap=max_gap,
        deploy_ok=runtime.get("deploy_ok", True),
        is_put=resolved_dir == TradeDirection.PUT if resolved_dir is not None else False,
    )
    indicator_cfg = params.get("indicators")
    if not isinstance(indicator_cfg, dict) or "windows" not in indicator_cfg:
        indicator_cfg = load_indicator_config_from_settings()
    squeeze_congestion = squeeze_congestion_active(
        prices,
        series,
        bb_window=int(indicator_cfg["windows"]["bb_window"]),
        bb_std_mult=float(indicator_cfg["multipliers"]["bb_std_mult"]),
        congestion=indicator_cfg["congestion"],
    )
    if squeeze_congestion:
        side_score = 0.51
    is_neutral_zone = calibration_mode == "neutral_zone" or resolved_dir is None
    entry = build_decision_entry(
        resolved_dir,
        calibrated_prob,
        execute=not is_neutral_zone,
        val_accuracy=val_accuracy,
        edge=calibrated_edge,
        train_loss=train_loss,
        trade_score=side_score if side_score is not None else 0.0,
        raw_prob=raw_prob,
        val_brier=float(runtime.get("val_brier", 1.0)),
        val_ece=float(runtime.get("val_ece", 1.0)),
        contract_duration=int(params.get("contract_duration", 180)),
    )
    _ = calibration_mode
    entry["metrics"]["gate_reason"] = "neutral_zone" if is_neutral_zone else None
    if is_neutral_zone:
        entry["metrics"]["signal_status"] = "SKIP:NEUTRAL_ZONE"
        entry["metrics"]["execution_candidate_ready"] = False
    entry["metrics"]["micro_chop_congestion"] = bool(squeeze_congestion)
    entry["metrics"]["edge_expectancy"] = None
    entry["metrics"]["calibrated_prob"] = calibrated_prob
    entry["metrics"]["calibration_mode"] = calibration_mode
    entry["metrics"]["calibrated_edge"] = calibrated_edge
    entry["metrics"]["cal_raw_gap_capped"] = bool(cal_capped)
    entry["metrics"]["cal_raw_gap"] = float(cal_raw_gap)
    entry["metrics"]["raw_margin"] = abs(raw_prob - 0.5)
    entry["metrics"]["cal_margin"] = abs(float(calibrated_prob) - 0.5)
    entry["metrics"]["direction_margin"] = abs(float(calibrated_prob) - 0.5)
    if calibrator is not None:
        entry["metrics"]["calibrator_method"] = str(getattr(calibrator, "method", "") or "")
        entry["metrics"]["calibrator_temperature"] = float(getattr(calibrator, "temperature", 1.0) or 1.0)
        entry["metrics"]["calibrator_platt_a"] = float(getattr(calibrator, "platt_a", 1.0) or 1.0)
        entry["metrics"]["calibrator_platt_b"] = float(getattr(calibrator, "platt_b", 0.0) or 0.0)
    entry["metrics"]["calibration_collapsed"] = bool(
        abs(raw_prob - 0.5) + 1e-12 >= 0.03 and abs(float(calibrated_prob) - 0.5) + 1e-12 < 0.03
    )
    entry["metrics"]["trend_direction"] = trend_dir.name
    entry["metrics"]["trend_type"] = trend_type
    entry["metrics"]["trend_period"] = trend_period
    entry["metrics"]["call_votes"] = call_votes
    entry["metrics"]["put_votes"] = put_votes
    entry["metrics"]["indicators"] = indicators_data
    if len(series.get("log_return", [])) > 0:
        idx = len(series["log_return"]) - 1
        entry["metrics"]["feature_vector"] = build_feature_row(series, idx).tolist()
    entry["metrics"]["indicator_timeframe_seconds"] = int(params.get("granularity", 3600))
    stamp_micro_frame_telemetry(_orch, str(symbol), entry["metrics"], params)
    if not isinstance(entry["metrics"].get("macro_indicators"), dict):
        entry["metrics"]["macro_indicators"] = indicators_data
    attach_dynamic_metrics(
        entry["metrics"],
        dynamic=dynamic,
        bb_width=bb_width,
        vol_ratio=vol_ratio,
        implied_vol_ratio=implied_vol_ratio,
        symbol=str(symbol),
        bb_history=_series_tail("bb_width", series),
        scale_enabled=bool(dynamic_cfg.get("implied_vol_bb_scale", True)),
        runtime=runtime,
    )
    return entry


__all__ = (
    "apply_calibration_neutral_tolerance",
    "build_prediction_context",
    "build_prediction_entry",
    "eager_local_predict",
    "prepare_meta_classifier_cross_symbol_bundle",
    "stamp_micro_frame_telemetry",
)
