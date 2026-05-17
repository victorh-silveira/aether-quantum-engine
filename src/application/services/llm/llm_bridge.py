"""Monta contexto M15, M5 e M3 e consulta a LLM para direcao por simbolo."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.llm.context_runtime import fetch_context_blocks, resolve_llm_runtime
from src.application.services.llm.duration_logic import enforce_minimum_duration
from src.application.services.llm.llm_bridge_debug import emit_direction_debug as _emit_direction_debug
from src.application.services.llm.llm_bridge_telemetry import (
    build_metrics_for_decision as _build_metrics_for_decision_core,
    store_symbol_decision,
)
from src.application.services.llm.llm_repeat_guard import (
    choose_direction_without_wait as _choose_direction_without_wait,
)
from src.application.services.llm.llm_symbol_io import (
    last_reference_price as _last_reference_price,
    request_llm_payload as _request_payload_core,
)
from src.application.services.llm.symbol_decision import _decision_from_payload, collect_symbol_llm_decision
from src.domain.models.trade import TradeDirection


_BRIDGE_TEST_REEXPORTS = (
    fetch_context_blocks,
    _decision_from_payload,
    _last_reference_price,
    _build_metrics_for_decision_core,
    _choose_direction_without_wait,
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

    # Configuração de correlação
    corr_cfg = orch.config.get("strategy", {}).get("correlation", {})
    corr_enabled = corr_cfg.get("enabled", False)
    anchor_sym = orch.anchor
    targets = corr_cfg.get("targets", {})

    budget = float(runtime["max_decision_latency_seconds"])
    cid = f"C{int(orch._active_cycle_id):04d}"

    # 1. Obter decisão da ÂNCORA
    try:
        orch.logger.debug("[%s] LLM_CORR || Consultando âncora: %s", cid, anchor_sym)
        direction, metrics = await asyncio.wait_for(
            _collect_symbol_decision(orch, sym=anchor_sym, runtime=runtime),
            timeout=budget,
        )
    except TimeoutError:
        orch.logger.warning("[%s] LLM_DEADLINE || âncora=%s || Falha total (Sem decisão Gemini)", cid, anchor_sym)
        # FAIL-SAFE ABSOLUTO: Se der timeout total, força uma direção para não ficar IDLE
        direction = TradeDirection.CALL
        metrics = llm_metrics(direction, 0.56, "FORCED EXEC (LLM Timeout)")

    # Inversão Global de Direção
    if direction is not None and orch.config.get("llm", {}).get("invert_llm_direction"):
        direction = TradeDirection.PUT if direction == TradeDirection.CALL else TradeDirection.CALL
        metrics = llm_metrics(direction, metrics["conviction"], f"INVERTED | {metrics.get('llm_note', '')}")
        orch.logger.debug("[%s] LLM_INVERT || Sinal da âncora invertido globalmente para %s", cid, direction.name)

    store_symbol_decision(decisions, anchor_sym, direction, metrics)
    orch._last_anchor_metrics = metrics

    # 2. Propagar sinal para os demais ativos via Correlação
    if direction is not None and corr_enabled:
        propagated_tags = []
        for target_sym in orch.symbols:
            if target_sym == anchor_sym:
                continue

            coeff = targets.get(target_sym, 1.0)
            # Propagação dinâmica baseada em Clusters Gemini
            us_targets = ("OTC_SPC", "OTC_NDX", "OTC_DJI")
            eu_targets = ("OTC_FCHI", "OTC_GDAXI", "OTC_SSMI", "OTC_FTSE")

            target_direction = direction
            if target_sym in us_targets and metrics.get("us_cluster"):
                target_direction = TradeDirection[metrics["us_cluster"]]
            elif target_sym in eu_targets and metrics.get("eu_cluster"):
                target_direction = TradeDirection[metrics["eu_cluster"]]
            elif coeff < 0:
                target_direction = TradeDirection.PUT if direction == TradeDirection.CALL else TradeDirection.CALL

            target_metrics = metrics.copy()
            target_metrics["llm_note"] = f"CLUSTER ({target_direction.name}) from {anchor_sym}"
            target_metrics["decision_source"] = "cluster_regime"

            # Ajuste de duracao minima por simbolo
            target_metrics["duration"] = enforce_minimum_duration(target_sym, target_metrics.get("duration", 15))

            store_symbol_decision(decisions, target_sym, target_direction, target_metrics)
            propagated_tags.append(f"{target_sym}[{target_direction.name[:1]}]")

        if propagated_tags:
            orch.logger.debug(
                "[%s] CORR CLUSTER || %s [%s] >> [%s]",
                cid,
                anchor_sym,
                direction.name,
                ", ".join(propagated_tags),
            )

    return decisions
