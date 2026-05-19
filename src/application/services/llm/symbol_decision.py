"""Pipeline de contexto, prompt e decisao LLM por simbolo (Analyse Profunda)."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.application.services.llm.context_runtime import runtime_for_gemini_call
from src.application.services.llm.duration_logic import calculate_adaptive_duration, enforce_minimum_duration
from src.application.services.llm.llm_bridge_debug import emit_direction_debug as _emit_direction_debug
from src.application.services.llm.llm_bridge_telemetry import (
    attach_decision_telemetry,
    build_metrics_for_decision as _build_metrics_for_decision_core,
    emit_llm_decision_log,
    emit_llm_http_snapshot,
    format_llm_runtime_thresholds,
)
from src.application.services.llm.llm_symbol_io import last_reference_price, request_llm_payload
from src.application.services.llm.prompt_utils import (
    iter_llm_prompt_audit_sections,
)
from src.application.services.llm.regime import _shannon_entropy
from src.application.services.llm.symbol_decision_utils import build_symbol_prompt
from src.domain.models.trade import TradeDirection


_request_payload = request_llm_payload


def _decision_from_payload(
    payload: dict[str, Any],
) -> tuple[TradeDirection | None, float, str, TradeDirection | None, TradeDirection | None]:
    """Converte payload padronizado em direcao, conviccao e nota."""
    tag = payload.get("_direction_normalized")
    note = " ".join(str(payload.get("note", "")).replace("\n", " ").split()).strip() or "-"

    if tag == "CALL":
        direction = TradeDirection.CALL
    elif tag == "PUT":
        direction = TradeDirection.PUT
    else:
        direction = None

    conviction = float(payload.get("_conviction_normalized", 0.0))

    us_dir_str = str(payload.get("us_cluster", "")).upper()
    eu_dir_str = str(payload.get("eu_cluster", "")).upper()

    us_dir = TradeDirection.CALL if us_dir_str == "CALL" else (TradeDirection.PUT if us_dir_str == "PUT" else direction)
    eu_dir = TradeDirection.CALL if eu_dir_str == "CALL" else (TradeDirection.PUT if eu_dir_str == "PUT" else direction)

    return direction, conviction, note, us_dir, eu_dir


async def collect_symbol_llm_decision(
    orch: Any,
    *,
    sym: str,
    runtime: dict[str, Any],
    llm_metrics: Any,
) -> tuple[TradeDirection | None, dict[str, Any]]:
    """Orquestra prompt, chamada Gemini e metricas por simbolo (Sempre Modo Profundo)."""
    (
        prompt,
        ctx,
        sniper_tok,
        baseline_prob,
        indicators_numeric_line,
        institutional_pa_bundle,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        mtf_d,
        swing_c,
    ) = await build_symbol_prompt(orch, sym, runtime)

    call_runtime, _ = runtime_for_gemini_call(runtime, ctx)
    cfg_root = getattr(orch, "config", None)
    raw_llm = cfg_root.get("llm") if isinstance(cfg_root, dict) else None
    symbols = list(getattr(orch, "symbols", ()) or ())
    leading_cycle_blank = int(orch._active_cycle_id) > 1 and (not symbols or sym == symbols[0])
    emit_llm_http_snapshot(
        orch.logger,
        sym,
        cycle_id=orch._active_cycle_id,
        http_user=prompt,
        http_system=str(runtime.get("llm_system") or ""),
        sniper_tokens=dict(sniper_tok),
        llm_config=raw_llm if isinstance(raw_llm, dict) else {},
        leading_cycle_blank=leading_cycle_blank,
    )
    ref_px = last_reference_price(orch.stream, sym)
    payload = await _request_payload(
        orch,
        sym,
        call_runtime,
        prompt,
        system=str(runtime.get("llm_system") or ""),
        cycle_id=int(orch._active_cycle_id),
    )
    llm_http_ms = float(payload.pop("_llm_latency_ms", 0) or 0)
    llm_resp_chars = int(payload.pop("_llm_raw_chars", 0) or 0)
    llm_direction_from_api = bool(payload.pop("_llm_direction_from_api", False))
    mtf_effective = str(mtf_d or "").strip() or "-"
    direction, conviction, note, us_dir, eu_dir = _decision_from_payload(payload)

    try:
        ic = runtime.get("indicator_config")
        if ic and swing_c:
            arr = np.array(swing_c, dtype=np.float64)
            ebins = int(getattr(ic, "entropy_bins", 30)) if hasattr(ic, "entropy_bins") else 30
            ewin = int(getattr(ic, "entropy_window", 20)) if hasattr(ic, "entropy_window") else 20
            entropy_val = _shannon_entropy(arr, ebins, ewin)
            if entropy_val > 3.0 and conviction > 0.75:
                conviction = 0.75  # pragma: no cover
                note += f" [ENTROPY_CAP: {entropy_val:.2f}]"  # pragma: no cover
    except Exception as e:  # pragma: no cover
        orch.logger.debug(f"Erro na trava de entropia: {e}")  # pragma: no cover

    if direction is None:
        z_val = float(ctx.get("zscore_value", 0.0))
        direction = TradeDirection.PUT if z_val > 0 else TradeDirection.CALL
        conviction = 0.56
        note = f"FORCED EXEC (LLM Refused): Z={z_val:.2f}"

    inv_threshold = float(runtime.get("inversion_threshold", 0.0))
    fol_threshold = float(runtime.get("follow_threshold", 0.0)) or inv_threshold

    inverted = False

    if direction:
        if conviction < inv_threshold:
            inverted = True
            direction = TradeDirection.PUT if direction == TradeDirection.CALL else TradeDirection.CALL
            note = f"Inverted: Conviction {conviction:.2f} < {inv_threshold:.2f}"
        elif conviction < fol_threshold:
            note = f"Follow (Noise Zone): {conviction:.2f}"

    direction, metrics = _build_metrics_for_decision_core(
        runtime,
        direction,
        conviction,
        note,
        ref_px,
        mtf_effective,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        llm_metrics,
        closes_m5=swing_c,
    )
    metrics["duration"] = enforce_minimum_duration(
        sym, calculate_adaptive_duration(runtime, ctx, base_duration=runtime.get("duration", 1))
    )
    attach_decision_telemetry(
        metrics,
        ctx,
        str(ctx.get("regime_label", "range")),
        baseline_prob,
        str(metrics.get("decision_source", "llm")),
        orch,
        sym,
    )
    _emit_direction_debug(
        orch,
        runtime=runtime,
        direction_base=direction,
        direction_final=direction,
        conviction=float(metrics.get("conviction", conviction)),
        macro_desc=macro_d,
        structure_desc=struct_d,
        swing_desc=swing_d,
        trigger_desc=trig_d,
        mtf_alignment=mtf_effective,
        decision_source=str(metrics.get("decision_source", "")),
        adjusted=inverted,
        exec_inverted=inverted,
    )
    metrics.update(
        {
            "execute": bool(metrics.get("execute", False)),
            "llm_exec_inverted": inverted,
            "llm_http_ms": llm_http_ms,
            "llm_response_chars": llm_resp_chars,
            "llm_direction_from_api": llm_direction_from_api,
            "us_cluster": us_dir.name if us_dir else None,
            "eu_cluster": eu_dir.name if eu_dir else None,
            "entry_policy_tag": "",
        }
    )
    audit_sections = iter_llm_prompt_audit_sections(
        sym,
        macro_d,
        struct_d,
        swing_d,
        trig_d,
        sniper_tok,
        mtf_d,
        str(ctx.get("regime_line", "")),
        str(ctx.get("session_line", "")),
        str(ctx.get("micro_line", "")),
        ctx.get("atr_m5_pct"),
        list(ctx.get("llm_trigger_closes") or []),
        runtime["payout_estimate"],
        runtime["min_payout_accept"],
        runtime["duration"],
        runtime["du"],
        trigger_ohlc=ctx.get("llm_trigger_ohlc"),
        strategy_payload=runtime.get("strategy_payload"),
        institutional_pa_bundle=institutional_pa_bundle,
        indicator_bundle_line=str(ctx.get("llm_indicator_bundle") or ""),
        wr_rolling=metrics.get("wr_rolling"),
        wr_samples=int(metrics.get("wr_samples") or 0),
    )
    emit_llm_decision_log(
        orch.logger,
        sym,
        cycle_id=orch._active_cycle_id,
        logic_line_max_chars=int(runtime.get("logic_line_max_chars", 140)),
        direction=direction,
        conviction=float(metrics.get("conviction", conviction)),
        ref_px=ref_px,
        model=runtime["model"],
        mtf_alignment=mtf_effective,
        justification=note,
        regime_label=str(ctx.get("regime_label", "range")),
        atr_m5_pct=ctx.get("atr_m5_pct"),
        baseline_prob=float(baseline_prob),
        wr_rolling=metrics.get("wr_rolling"),
        wr_samples=int(metrics.get("wr_samples") or 0),
        decision_source=str(metrics.get("decision_source", "llm")),
        indicator_cfg=str(ctx.get("llm_indicator_cfg") or ""),
        indicators_numeric_line=indicators_numeric_line,
        runtime_thresholds=format_llm_runtime_thresholds(runtime),
        prompt_char_count=len(prompt),
        prompt_audit_sections=audit_sections,
        engine_runtime=runtime,
        motor_note=note if inverted else "",
        llm_http_ms=llm_http_ms,
        llm_response_chars=llm_resp_chars,
        llm_direction_from_api=llm_direction_from_api,
        us_cluster=us_dir.name if us_dir else None,
        eu_cluster=eu_dir.name if eu_dir else None,
    )
    return direction, metrics
