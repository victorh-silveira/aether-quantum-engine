"""Montagem compartilhada de entrada de decisao DL."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_bridge_helpers import build_decision_entry
from src.application.services.deep_learning.dl_calibration import CalibratorState, calibrate_trade_score
from src.application.services.deep_learning.dl_feature_build import build_feature_row, precompute_price_series
from src.application.services.deep_learning.dl_gating import resolve_calibrated_edge, resolve_confidence_thresholds
from src.application.services.deep_learning.dl_params import parse_dynamic_threshold_config
from src.application.services.deep_learning.dl_predict_metrics import attach_dynamic_metrics
from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model
from src.application.services.deep_learning.dl_trend import calculate_trend_direction
from src.application.services.deep_learning.model import predict_next_direction
from src.application.services.execution_volatility_threshold import resolve_dynamic_threshold_bundle
from src.application.services.meta_classifier_cross_symbol import attach_cross_symbol_features_to_decisions
from src.application.services.meta_classifier_flow_features import flow_features_from_micro_series
from src.domain.models.trade import TradeDirection


def _series_tail(series_key: str, series: dict) -> list[float]:
    """Extrai historico numerico de uma serie de indicadores."""
    chunk = series.get(series_key)
    if chunk is None or len(chunk) == 0:
        return []
    return [float(v) for v in chunk]


def _infer_direction(calibrated_prob: float, direction: TradeDirection | None, pivot: float = 0.5) -> TradeDirection:
    """Infere CALL ou PUT a partir da probabilidade calibrada quando direction e None."""
    if direction is not None:
        return direction
    return TradeDirection.CALL if float(calibrated_prob) > float(pivot) else TradeDirection.PUT


def stamp_micro_frame_telemetry(orch: Any, symbol: str, metrics: dict[str, Any], params: dict[str, Any]) -> None:
    """Anexa telemetria micro M1, fluxo de ticks e desvio Keltner para meta-classificador."""
    stream = getattr(orch, "stream", None)
    if stream is None or not hasattr(stream, "get_micro_numpy_series"):
        return
    closes = stream.get_micro_numpy_series(str(symbol), "close")
    if closes is None or len(closes) < 8:
        return
    micro_gran = int(params.get("micro_granularity", 60))
    high = stream.get_micro_numpy_series(str(symbol), "high")
    low = stream.get_micro_numpy_series(str(symbol), "low")
    open_ = stream.get_micro_numpy_series(str(symbol), "open")
    series = precompute_price_series(closes, granularity=micro_gran, symbol=str(symbol))
    rsi = float(series["rsi"][-1]) if len(series.get("rsi", [])) > 0 else 0.0
    vol_ratio = float(series["vol_ratio_short_long"][-1]) if len(series.get("vol_ratio_short_long", [])) > 0 else 0.0
    metrics["micro_indicators"] = {"rsi": rsi, "vol_ratio": vol_ratio}
    flow = flow_features_from_micro_series(
        closes,
        granularity=micro_gran,
        symbol=str(symbol),
        open_=open_,
        high=high,
        low=low,
    )
    tick_buffer = getattr(stream, "tick_buffer", None)
    if tick_buffer is not None and hasattr(tick_buffer, "live_tick_acceleration"):
        flow["micro_tick_acceleration"] = float(tick_buffer.live_tick_acceleration(str(symbol)))
    metrics["flow_features"] = flow


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
    gran = int(granularity or params.get("granularity", 60))
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
    bb_width = float(series["bb_width"][-1]) if len(series.get("bb_width", [])) > 0 else 0.0
    atr_norm = float(series["atr_norm"][-1]) if len(series.get("atr_norm", [])) > 0 else 0.0
    adx = float(series["adx"][-1]) if len(series.get("adx", [])) > 0 else 0.0
    vol_ratio = float(series["vol_ratio_short_long"][-1]) if len(series.get("vol_ratio_short_long", [])) > 0 else 0.0
    implied_vol_ratio = float(series["implied_vol_ratio"][-1]) if len(series.get("implied_vol_ratio", [])) > 0 else 1.0
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
    bb_width = float(series["bb_width"][-1]) if len(series.get("bb_width", [])) > 0 else 0.0
    atr_norm = float(series["atr_norm"][-1]) if len(series.get("atr_norm", [])) > 0 else 0.0
    adx = float(series["adx"][-1]) if len(series.get("adx", [])) > 0 else 0.0
    vol_ratio = float(series["vol_ratio_short_long"][-1]) if len(series.get("vol_ratio_short_long", [])) > 0 else 0.0
    implied_vol_ratio = float(series["implied_vol_ratio"][-1]) if len(series.get("implied_vol_ratio", [])) > 0 else 1.0
    trend_dir, trend_type, trend_period, call_votes, put_votes = calculate_trend_direction(prices, series, exec_cfg)
    indicators_data = {
        "hurst": float(series["hurst"][-1]) if "hurst" in series and len(series["hurst"]) > 0 else 0.0,
        "adx": adx,
        "vol_ratio": vol_ratio,
        "implied_vol_ratio": implied_vol_ratio,
        "bb_width": bb_width,
        "atr_norm": atr_norm,
        "cmo": float(series["cmo"][-1]) if "cmo" in series and len(series["cmo"]) > 0 else 0.0,
        "keltner": float(series["keltner_pct_b"][-1])
        if "keltner_pct_b" in series and len(series["keltner_pct_b"]) > 0
        else 0.0,
        "bb_pct_b": float(series["bb_pct_b"][-1]) if "bb_pct_b" in series and len(series["bb_pct_b"]) > 0 else 0.5,
        "rsi": float(series["rsi"][-1]) if "rsi" in series and len(series["rsi"]) > 0 else 0.0,
        "macd": float(series["macd"][-1]) if "macd" in series and len(series["macd"]) > 0 else 0.0,
        "macd_sig": float(series["macd_signal"][-1])
        if "macd_signal" in series and len(series["macd_signal"]) > 0
        else 0.0,
        "di_diff": float(series["di_diff"][-1]) if "di_diff" in series and len(series["di_diff"]) > 0 else 0.0,
    }
    calibrated_prob = float(prob)
    pivot = (call_threshold + put_threshold) * 0.5
    resolved_dir = _infer_direction(calibrated_prob, direction, pivot=pivot)
    calibrated_edge = resolve_calibrated_edge(calibrated_prob, raw_prob=raw_prob)
    calibrator = runtime.get("calibrator")
    side_score = calibrate_trade_score(
        raw_prob,
        val_accuracy,
        calibrator,
        deploy_ok=runtime.get("deploy_ok", True),
        is_put=resolved_dir == TradeDirection.PUT,
    )
    entry = build_decision_entry(
        resolved_dir,
        prob,
        execute=True,
        val_accuracy=val_accuracy,
        edge=calibrated_edge,
        train_loss=train_loss,
        trade_score=side_score,
        raw_prob=raw_prob,
        val_brier=float(runtime.get("val_brier", 1.0)),
        val_ece=float(runtime.get("val_ece", 1.0)),
        contract_duration=int(params.get("contract_duration", 60)),
    )
    entry["metrics"]["gate_reason"] = None
    entry["metrics"]["edge_expectancy"] = None
    entry["metrics"]["calibrated_prob"] = calibrated_prob
    entry["metrics"]["calibrated_edge"] = calibrated_edge
    entry["metrics"]["trend_direction"] = trend_dir.name
    entry["metrics"]["trend_type"] = trend_type
    entry["metrics"]["trend_period"] = trend_period
    entry["metrics"]["call_votes"] = call_votes
    entry["metrics"]["put_votes"] = put_votes
    entry["metrics"]["indicators"] = indicators_data
    entry["metrics"]["macro_indicators"] = indicators_data
    if len(series.get("log_return", [])) > 0:
        idx = len(series["log_return"]) - 1
        entry["metrics"]["feature_vector"] = build_feature_row(series, idx).tolist()
    entry["metrics"]["indicator_timeframe_seconds"] = int(params.get("granularity", 900))
    stamp_micro_frame_telemetry(_orch, str(symbol), entry["metrics"], params)
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


def prepare_meta_classifier_cross_symbol_bundle(
    orch: Any,
    decisions: dict[str, dict],
    params: dict[str, Any],
) -> None:
    """Centraliza telemetria micro paralela e spreads cross-symbol antes do prefetch meta."""
    for symbol, entry in decisions.items():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        stamp_micro_frame_telemetry(orch, str(symbol), metrics, params)
    attach_cross_symbol_features_to_decisions(decisions)
