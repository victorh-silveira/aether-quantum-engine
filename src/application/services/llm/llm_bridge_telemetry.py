"""Utilitarios de telemetria e armazenamento para o bridge LLM."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def attach_llm_metadata(metrics: dict[str, Any], *, model: str, ref_px: float | None) -> dict[str, Any]:
    """Anexa modelo e preco de referencia as metricas de decisao."""
    metrics["model"] = model
    metrics["ref_px"] = ref_px
    return metrics


def emit_llm_decision_log(
    target_logger: logging.Logger,
    symbol: str,
    *,
    cycle_id: int | None,
    logic_line_max_chars: int,
    direction: TradeDirection,
    conviction: float,
    ref_px: float | None,
    model: str,
    mtf_alignment: str,
    justification: str,
    regime_label: str,
    atr_m5_pct: float | None,
    baseline_prob: float,
    wr_rolling: float | None,
    wr_samples: int,
    decision_source: str,
    indicator_cfg: str,
    indicators_numeric_line: str,
    runtime_thresholds: str,
    prompt_char_count: int,
    prompt_audit_sections: list[tuple[str, str]],
    motor_score_mtf: int = 0,
    motor_note: str = "",
    engine_runtime: dict[str, Any] | None = None,
    llm_http_ms: float = 0.0,
    llm_response_chars: int = 0,
    entry_policy_tag: str = "",
    llm_direction_from_api: bool = False,
    us_cluster: str | None = None,
    eu_cluster: str | None = None,
    macro_sentiment: str | None = None,
) -> None:
    """Emite logs estruturados de auditoria para a decisao da LLM."""
    _ = (
        logic_line_max_chars,
        justification,
        regime_label,
        atr_m5_pct,
        baseline_prob,
        wr_rolling,
        wr_samples,
        decision_source,
        indicator_cfg,
        runtime_thresholds,
        prompt_char_count,
        prompt_audit_sections,
        motor_score_mtf,
        motor_note,
        engine_runtime,
        entry_policy_tag,
    )
    cid = f"[C{int(cycle_id):04d}] " if cycle_id is not None else ""
    px = f"{ref_px:.5f}" if ref_px is not None else "n/a"
    gapi = "1" if llm_direction_from_api else "0"
    atr_val = f"{atr_m5_pct:.3f}%" if atr_m5_pct is not None else "n/a"

    target_logger.debug(f"{cid}LLM_AUDIT || {symbol} || reg={regime_label} || sigma={atr_val}")
    target_logger.debug(f"{cid}LLM_AUDIT || [NUM] {indicators_numeric_line}")

    target_logger.debug(
        f"{cid}LLM_DADOS || {symbol} || reg={regime_label} || prompt={prompt_char_count}ch resp={llm_response_chars}ch"
    )
    target_logger.debug(f"{cid}LLM_DADOS || [MTF] {mtf_alignment}")
    target_logger.debug(f"{cid}LLM_DADOS || [SRC] {decision_source}")
    res_tag = direction.name if direction else "FAIL"
    if not direction and decision_source == "llm_skip":
        res_tag = "SKIP"

    inv_flag = " [INV]" if "Inverted" in (motor_note or "") else ""

    us_c = us_cluster or "-"
    eu_c = eu_cluster or "-"
    macro_part = f" macro={macro_sentiment}" if macro_sentiment else ""
    cluster_str = f" US={us_c} EU={eu_c}{macro_part}"

    target_logger.info(
        f"{cid}LLM_RESPOSTA || {symbol} || [{res_tag}]{inv_flag}{cluster_str} prob={conviction:.1%} || ref={px} http_ms={int(llm_http_ms)} || gapi={gapi} model={model}"
    )


def format_llm_runtime_thresholds(runtime: dict[str, Any]) -> str:
    """Resume limites de decisao para log."""
    tout = float(runtime.get("timeout", 0))
    toks = int(runtime.get("num_predict", 0))
    mconv = float(runtime.get("min_conviction_execute", 0))
    return f"tout={tout:.1f}s toks={toks} mconv={mconv:.2f}"


def attach_decision_telemetry(
    metrics: dict[str, Any],
    _ctx: dict[str, Any],
    regime_label: str,
    baseline_prob: float,
    decision_source: str,
    orch: Any,
    symbol: str,
) -> None:
    """Insere dados de contexto e performance nas metricas de execucao."""
    metrics["regime_label"] = regime_label
    metrics["baseline_prob"] = baseline_prob
    metrics["decision_source"] = decision_source
    metrics["symbol"] = symbol
    metrics["cycle_id"] = int(orch._active_cycle_id)

    rm = getattr(orch, "risk_manager", None)
    if rm is not None and hasattr(rm, "get_wr_rolling_stats"):
        raw_wr = rm.get_wr_rolling_stats(symbol)
        if isinstance(raw_wr, tuple) and len(raw_wr) == 2:
            metrics["wr_rolling"] = raw_wr[0]
            metrics["wr_samples"] = int(raw_wr[1])


def store_symbol_decision(
    decisions: dict[str, dict], sym: str, direction: TradeDirection, metrics: dict[str, Any]
) -> None:
    """Armazena a decisao no mapa de execucao do ciclo."""
    decisions[sym] = {"direction": direction, "metrics": metrics}


def build_metrics_for_decision(
    runtime: dict[str, Any],
    direction: TradeDirection | None,
    conviction: float,
    note: str,
    ref_px: float | None,
    _mtf_alignment: str,
    _macro_d: str,
    _struct_d: str,
    _swing_d: str,
    _trig_d: str,
    llm_metrics_fn: Any,
    *,
    closes_m5: list[float] | None = None,
) -> tuple[TradeDirection | None, dict[str, Any]]:
    """Monta metricas finais transferindo soberania total para a LLM, respeitando o limite de convicção."""
    _ = (closes_m5, _mtf_alignment, _macro_d, _struct_d, _swing_d, _trig_d)
    metrics = attach_llm_metadata(llm_metrics_fn(direction, conviction, note), model=runtime["model"], ref_px=ref_px)

    min_conv = float(runtime.get("min_conviction_execute", 0.55))
    metrics["execute"] = (direction is not None) and (conviction >= min_conv)
    metrics["llm_regime"] = "selective_conviction"
    metrics["llm_direction_adjusted"] = False
    return direction, metrics


def _truncate_preview(text: str, cap: int | None) -> str:
    """Normaliza texto de preview e aplica limite opcional de caracteres."""
    raw = str(text or "").replace("\n", " ").strip()
    if cap is None:
        return raw
    limit = max(64, min(16000, int(cap)))
    return raw[:limit].rstrip() if len(raw) > limit else raw


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Acrescenta um registro JSONL ao arquivo de dump de IO LLM."""
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def emit_llm_http_snapshot(
    logger_obj: Any,
    symbol: str,
    *,
    cycle_id: int,
    http_user: str,
    http_system: str,
    sniper_tokens: dict[str, Any],
    llm_config: dict[str, Any] | None,
    leading_cycle_blank: bool = False,
    http_system_resolved: str = "",
    mtf_matrix: str = "",
    indicators_numeric_line: str = "",
    institutional_pa_bundle: str = "",
    indicator_bundle_line: str = "",
    tf_labels: tuple[str, ...] | list[str] = (),
    macro_confluence: str = "",
    fx_reference_line: str = "",
    macro_sentiment: str = "",
) -> None:
    """Registra em INFO o prompt usuario e grava JSON opcional do snapshot."""
    cfg = llm_config or {}
    cid = f"C{int(cycle_id):04d}"
    if leading_cycle_blank:
        logger_obj.info("")
    sys_resolved = (http_system_resolved or http_system or "").strip()
    if bool(cfg.get("log_llm_io_line", True)):
        raw_u = str(http_user or "").replace("\n", " ").strip()
        nu = len(raw_u)
        raw_sys = sys_resolved.replace("\n", " ").strip()
        nsys = len(raw_sys)
        cap = cfg.get("log_llm_io_preview_chars")
        preview_u = _truncate_preview(raw_u, cap)
        preview_s = _truncate_preview(raw_sys, cap)
        logger_obj.info(
            "[%s] LLM_IO || %s || user_ch=%s preview_user=%s",
            cid,
            symbol,
            nu,
            preview_u,
        )
        logger_obj.info(
            "[%s] LLM_IO || %s || sys_ch=%s preview_sys=%s",
            cid,
            symbol,
            nsys,
            preview_s,
        )
    raw_path = cfg.get("log_llm_io_dump_path")
    path_txt = str(raw_path).strip() if raw_path not in (None, False) else ""
    if not path_txt:
        return
    payload = {
        "cycle_id": int(cycle_id),
        "symbol": symbol,
        "http_user": str(http_user or ""),
        "http_system": str(http_system or ""),
        "http_system_resolved": sys_resolved,
        "sniper_tokens": dict(sniper_tokens),
        "mtf_matrix": str(mtf_matrix or ""),
        "indicators_numeric_line": str(indicators_numeric_line or ""),
        "institutional_pa_bundle": str(institutional_pa_bundle or ""),
        "indicator_bundle_line": str(indicator_bundle_line or ""),
        "tf_labels": list(tf_labels),
        "macro_confluence": str(macro_confluence or ""),
        "fx_reference_line": str(fx_reference_line or ""),
        "macro_sentiment": str(macro_sentiment or ""),
    }
    try:
        p = Path(path_txt)
        if not p.is_absolute():
            p = Path.cwd() / p
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix.lower() == ".jsonl":
            _append_jsonl(p, payload)
        else:
            p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        logger_obj.warning("[%s] LLM_IO_DUMP_FAIL path=%s", cid, path_txt)
