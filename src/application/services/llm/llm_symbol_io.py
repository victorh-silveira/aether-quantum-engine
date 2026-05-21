"""IO auxiliar para chamada LLM e referencia de preco por simbolo."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.application.services.llm.llm_bridge_utils import parse_llm_trade_response
from src.application.services.llm.llm_decision import get_decision


_logger = logging.getLogger("AETH")


def _sovereign_llm_payload(
    direction: str,
    *,
    conviction: float = 1.0,
    http_ms: float,
    raw_chars: int,
    llm_failed: bool,
    llm_direction_from_api: bool = False,
    us_cluster: str | None = None,
    eu_cluster: str | None = None,
) -> dict[str, Any]:
    """Payload metricas a partir do token devolvido pela API."""
    raw = (direction or "").strip().upper()
    d = raw if raw in ("CALL", "PUT") else None
    conv = float(conviction)
    note = "llm_timeout" if llm_failed else (d or "none")
    out: dict[str, Any] = {
        "direction": d,
        "conviction": conv,
        "note": note,
        "_direction_normalized": d,
        "_conviction_normalized": conv,
        "_llm_latency_ms": round(http_ms, 1),
        "_llm_raw_chars": raw_chars,
        "_llm_direction_from_api": bool(llm_direction_from_api),
    }
    if us_cluster:
        out["us_cluster"] = us_cluster
    if eu_cluster:
        out["eu_cluster"] = eu_cluster
    if llm_failed:
        out["_llm_call_failed"] = True
    return out


def _cycle_log_prefix(cycle_id: int | None) -> str:
    """Retorna prefixo [Cnnnn] para logs de IO ou string vazia."""
    if cycle_id is None:
        return ""
    return f"[C{int(cycle_id):04d}] "


async def request_llm_payload(
    _orch: Any,
    sym: str,
    runtime: dict[str, Any],
    prompt: str,
    *,
    system: str,
    cycle_id: int | None = None,
) -> dict[str, Any]:
    """Consulta Gemini; CALL ou PUT somente quando a API devolver texto parseavel."""
    base_url = runtime["base_url"]
    model = runtime["model"]
    timeout = runtime["timeout"]
    num_predict = runtime["num_predict"]
    soft_cap = float(runtime.get("llm_async_outer_seconds", float(timeout) + max(2.0, 5.0)))
    gen_cfg = runtime.get("generation_config")
    generation_config = gen_cfg if isinstance(gen_cfg, dict) else None
    api_key = str(runtime.get("gemini_api_key") or "").strip()
    lp = _cycle_log_prefix(cycle_id)
    t0 = time.perf_counter()
    try:
        dir_raw, from_api, raw_text = await asyncio.wait_for(
            get_decision(
                base_url,
                model,
                prompt,
                timeout,
                system=system,
                num_predict=num_predict,
                temperature=float(runtime.get("llm_temperature", 0.0)),
                generation_config=generation_config,
                api_key=api_key or None,
                safety_retry_attempts=int(runtime.get("llm_retry_attempts", 1)),
                log_cycle_id=cycle_id,
            ),
            timeout=soft_cap,
        )
    except TimeoutError:
        http_ms = (time.perf_counter() - t0) * 1000.0
        _logger.debug(
            "%sLLM Gemini http_ms=%.0f asyncio_timeout ref_gemini_2_5_flash_tipico~300",
            lp,
            http_ms,
        )
        _logger.info(
            "%sLLM Gemini: timeout asyncio externo; ignorando (Preservação de Capital)",
            lp,
        )
        return _sovereign_llm_payload(
            "",
            http_ms=http_ms,
            raw_chars=0,
            llm_failed=True,
            llm_direction_from_api=False,
        )
    http_ms = (time.perf_counter() - t0) * 1000.0
    _logger.debug(
        "%sLLM Gemini http_ms=%.0f ref_gemini_2_5_flash_tipico~300 simbolo=%s",
        lp,
        http_ms,
        sym,
    )
    if http_ms > 450.0:
        _logger.debug(
            "%sLLM Gemini latencia elevada http_ms=%.0f simbolo=%s",
            lp,
            http_ms,
            sym,
        )
    raw_len = len(str(raw_text or ""))
    conv_val = 1.0
    us_c = None
    eu_c = None
    if raw_text:
        parsed = parse_llm_trade_response(raw_text)
        conv_val = float(parsed.get("conviction", 1.0))
        us_c = parsed.get("us_cluster")
        eu_c = parsed.get("eu_cluster")

    return _sovereign_llm_payload(
        str(dir_raw or ""),
        conviction=conv_val,
        http_ms=http_ms,
        raw_chars=raw_len,
        llm_failed=False,
        llm_direction_from_api=bool(from_api),
        us_cluster=us_c,
        eu_cluster=eu_c,
    )


def last_reference_price(stream: Any, symbol: str) -> float | None:
    """Ultimo fechamento do buffer para referencia no log."""
    try:
        series = stream.get_numpy_series(symbol, field="close")
        if series is None or len(series) == 0:
            return None
        return float(series[-1])
    except Exception:
        return None
