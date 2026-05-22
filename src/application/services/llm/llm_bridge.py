"""Monta contexto M15, M5 e M3 e consulta a LLM para direcao por simbolo."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.llm.context_runtime import fetch_context_blocks, resolve_llm_runtime
from src.application.services.llm.llm_bridge_debug import emit_direction_debug as _emit_direction_debug
from src.application.services.llm.llm_bridge_telemetry import (
    build_metrics_for_decision as _build_metrics_for_decision_core,
    store_symbol_decision,
)
from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
from src.application.services.llm.llm_symbol_io import (
    last_reference_price as _last_reference_price,
    request_llm_payload as _request_payload_core,
)
from src.application.services.llm.symbol_decision import collect_symbol_llm_decision
from src.application.services.llm.symbol_decision_utils import decision_from_payload as _decision_from_payload
from src.domain.models.trade import TradeDirection


_BRIDGE_TEST_REEXPORTS = (
    fetch_context_blocks,
    _decision_from_payload,
    _last_reference_price,
    _build_metrics_for_decision_core,
    _emit_direction_debug,
)


def llm_metrics(direction: TradeDirection | None, conviction: float, note: str) -> dict[str, Any]:
    """Metricas compativeis com o ExecutionManager e logs."""
    note_clean = (note or "").replace("\n", " ").strip()[:120]
    if direction is None:
        src = "llm_skip" if "SKIP" in note_clean.upper() else "llm_api_failure"
        return {
            "conviction": 0.0,
            "direction": "NONE",
            "execute": False,
            "llm_note": note_clean if note_clean else "llm_no_response",
            "decision_source": src,
            "prob_call": 0.5,
            "prob_put": 0.5,
        }

    is_call = direction == TradeDirection.CALL
    pc = 1.0 if is_call else 0.0
    macro_bias = 1.0 if is_call else -1.0
    call_score = 1.0 if is_call else 0.0
    put_score = 0.0 if is_call else 1.0

    note_clean = (note or "").replace("\n", " ").strip()[:120]
    return {
        "conviction": conviction,
        "direction": direction.name,
        "execute": True,
        "llm_note": note_clean,
        "decision_source": "llm",
        "macro_bias": macro_bias,
        "pattern_tags": ["LLM", note_clean[:80]] if note_clean else ["LLM"],
        "price": 0.0,
        "call_score": call_score,
        "put_score": put_score,
        "prob_call": pc,
        "prob_put": 1.0 - pc,
        "h1_trend": 0.0,
        "d1_trend": 0.0,
        "macro_slope": 0.0,
        "mtf_structure_bull_n": 0,
        "mtf_structure_bear_n": 0,
    }


async def _request_payload(
    orch: Any,
    sym: str,
    runtime: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    """Consulta LLM e aplica parse fail-safe para payload padronizado."""
    return await _request_payload_core(
        orch,
        sym,
        runtime,
        prompt,
        system=str(runtime.get("llm_system") or ""),
    )


async def _collect_symbol_decision(
    orch: Any,
    *,
    sym: str,
    runtime: dict[str, Any],
) -> tuple[TradeDirection | None, dict[str, Any]]:
    """Processa a decisao de um simbolo com parse e filtros de seguranca."""
    return await collect_symbol_llm_decision(
        orch,
        sym=sym,
        runtime=runtime,
        llm_metrics=llm_metrics,
    )


async def collect_llm_decisions(orch: Any) -> dict[str, dict]:
    """Produz mapa symbol -> {direction, metrics} usando EXCLUSIVAMENTE Gemini para a âncora."""
    runtime = resolve_llm_runtime(orch)
    decisions: dict[str, dict] = {}

    corr_cfg = orch.config.get("strategy", {}).get("correlation", {})
    cluster_propagation_enabled = bool(corr_cfg.get("enabled", False))
    anchor_sym = orch.anchor

    budget = float(runtime["max_decision_latency_seconds"])
    cid = f"C{int(orch._active_cycle_id):04d}"

    try:
        orch.logger.debug("[%s] LLM_CORR || Consultando âncora: %s", cid, anchor_sym)
        direction, metrics = await asyncio.wait_for(
            _collect_symbol_decision(orch, sym=anchor_sym, runtime=runtime),
            timeout=budget,
        )
    except TimeoutError:
        orch.logger.warning(
            "[%s] LLM_DEADLINE || âncora=%s || Falha total (Timeout de Decisão Gemini)", cid, anchor_sym
        )
        direction = None
        metrics = llm_metrics(direction, 0.0, "LLM Timeout (Capital Preserved)")

    store_symbol_decision(decisions, anchor_sym, direction, metrics)
    orch._last_anchor_metrics = metrics

    if direction is not None and cluster_propagation_enabled:
        propagate_cluster_decisions(
            orch,
            anchor_sym=anchor_sym,
            direction=direction,
            metrics=metrics,
            decisions=decisions,
            cid=cid,
        )

    return decisions
