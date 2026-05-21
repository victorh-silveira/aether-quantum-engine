"""Chamadas ao Google Gemini (SDK google-genai) para decisao soberana via LLM pura."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.application.services.llm.gemini_constants import (
    GEMINI_DEFAULT_MODEL,
    GEMINI_FALLBACK_MODEL,
    GEMINI_HTTP_CEILING_SEC,
)
from src.application.services.llm.llm_bridge_utils import parse_llm_trade_response
from src.application.services.llm.response_extract import (
    extract_llm_text,
    llm_default_safety_settings,
    log_llm_empty_response,
)
from src.application.services.llm.sovereign_system import SOVEREIGN_SYSTEM


logger = logging.getLogger("AETH")

_GEMINI_RETRY_TAIL = (
    "\n\nResponda OBRIGATORIAMENTE no formato exigido: "
    "EURUSD: CALL ou PUT | US_CLUSTER: CALL ou PUT | EU_CLUSTER: CALL ou PUT | Probabilidade: [0.XX]. "
    "Proibido WAIT, SKIP, vazio ou omitir US_CLUSTER/EU_CLUSTER."
)


def resolved_system_instruction(runtime_system: str | None) -> str:
    """Junta a instrucao de sistema fixa com texto opcional de configuracao."""
    extra = (runtime_system or "").strip()
    if not extra:
        return SOVEREIGN_SYSTEM
    return f"{SOVEREIGN_SYSTEM}\n\n{extra}"


def _resolved_system_instruction(runtime_system: str | None) -> str:
    """Alias interno para resolved_system_instruction."""
    return resolved_system_instruction(runtime_system)


def _cycle_prefix(log_cycle_id: int | None) -> str:
    """Retorna prefixo [Cnnnn] para logs Gemini ou string vazia."""
    if log_cycle_id is None:
        return ""
    return f"[C{int(log_cycle_id):04d}] "


def is_retryable_gemini_error(exc: BaseException) -> bool:
    """Indica erro transitorio da API Gemini (timeout, sobrecarga ou cancelamento)."""
    msg = str(exc).upper()
    markers = (
        "DEADLINE_EXCEEDED",
        "504",
        "503",
        "UNAVAILABLE",
        "499",
        "CANCELLED",
        "TIMEOUT",
        "RESOURCE_EXHAUSTED",
        "429",
    )
    return any(m in msg for m in markers)


def _merge_generation_config(
    base_temperature: float,
    num_predict: int | None,
    extra: dict[str, Any] | None,
    *,
    system_instruction: str,
) -> Any:
    """Constroi GenerateContentConfig mesclando temperatura e limites de saida."""
    tokens = 48
    if num_predict is not None:
        tokens = max(8, min(4096, int(num_predict)))
    cfg_base: dict[str, Any] = {
        "temperature": base_temperature,
        "max_output_tokens": tokens,
        "system_instruction": system_instruction,
        "safety_settings": llm_default_safety_settings(types),
    }
    if extra:
        for k, v in extra.items():
            if k in ("safety_settings", "system_instruction"):
                continue
            if k == "thinking_config" and isinstance(v, dict):
                budget = int(v.get("thinking_budget", 0))
                cfg_base["thinking_config"] = types.ThinkingConfig(thinking_budget=budget)
            else:
                cfg_base[k] = v
    return types.GenerateContentConfig(**cfg_base)


def _sync_generate(model_name: str, api_key: str, prompt: str, gen_cfg: Any, deadline_sec: float) -> str:
    """Executa generate_content de forma sincrona para uso em asyncio.to_thread."""
    timeout_ms = int(max(1000.0, min(deadline_sec, GEMINI_HTTP_CEILING_SEC) * 1000.0))
    http_opts = types.HttpOptions(timeout=timeout_ms)
    client = genai.Client(api_key=api_key, http_options=http_opts)
    resp = client.models.generate_content(model=model_name, contents=prompt, config=gen_cfg)
    out = extract_llm_text(resp)
    if not out:
        log_llm_empty_response(resp, logger)
    return out


async def _attempt_generate(
    *,
    model_name: str,
    api_key: str,
    body: str,
    gen_cfg: Any,
    deadline: float,
    log_cycle_id: int | None,
) -> tuple[str | None, bool, str, BaseException | None]:
    """Uma tentativa de geracao; retorna direcao normalizada ou erro."""
    try:
        raw = await asyncio.wait_for(
            asyncio.to_thread(
                _sync_generate,
                model_name,
                api_key,
                body,
                gen_cfg,
                deadline,
            ),
            timeout=deadline + 1.0,
        )
        parsed = parse_llm_trade_response(raw)
        norm = parsed.get("direction")
        if norm in ("CALL", "PUT"):
            return (norm, True, raw, None)
        logger.info(
            "%sLLM: Resposta inesperada (nao normalizou): '%s'",
            _cycle_prefix(log_cycle_id),
            raw[:100],
        )
        return (None, False, raw, None)
    except Exception as exc:
        return (None, False, "", exc)


async def get_decision(
    _base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    *,
    system: str | None = None,
    num_predict: int | None = None,
    temperature: float = 0.0,
    generation_config: dict[str, Any] | None = None,
    api_key: str | None = None,
    safety_retry_attempts: int = 0,
    log_cycle_id: int | None = None,
    fallback_model: str | None = None,
) -> tuple[str | None, bool, str]:
    """Obtem decisao EXCLUSIVAMENTE da LLM via Google Gemini SDK."""
    load_dotenv()
    key = api_key or os.getenv("GEMINI_API_KEY")
    if not key:
        logger.error("%sLLM: API Key ausente", _cycle_prefix(log_cycle_id))
        return (None, False, "")

    sys_inst = _resolved_system_instruction(system)
    gen_cfg = _merge_generation_config(temperature, num_predict, generation_config, system_instruction=sys_inst)
    primary = (model or GEMINI_DEFAULT_MODEL).strip() or GEMINI_DEFAULT_MODEL
    fallback = (fallback_model or GEMINI_FALLBACK_MODEL).strip() or GEMINI_FALLBACK_MODEL
    active_model = primary
    deadline = min(float(timeout or 15.0), GEMINI_HTTP_CEILING_SEC)

    total_tries = 1 + max(0, int(safety_retry_attempts))
    reminder = ""
    switched_fallback = False

    for attempt in range(total_tries):
        body = f"{prompt}{reminder}"
        norm, from_api, raw, err = await _attempt_generate(
            model_name=active_model,
            api_key=key,
            body=body,
            gen_cfg=gen_cfg,
            deadline=deadline,
            log_cycle_id=log_cycle_id,
        )
        if norm in ("CALL", "PUT"):
            return (norm, from_api, raw)

        if err is not None:
            logger.info(
                "%sLLM: Tentativa %d (%s) falhou: %s",
                _cycle_prefix(log_cycle_id),
                attempt + 1,
                active_model,
                err,
            )
            if not switched_fallback and active_model != fallback and is_retryable_gemini_error(err):
                active_model = fallback
                switched_fallback = True
                logger.info(
                    "%sLLM: Alternando para modelo fallback %s",
                    _cycle_prefix(log_cycle_id),
                    fallback,
                )

        reminder = _GEMINI_RETRY_TAIL
        if attempt + 1 < total_tries:
            backoff = min(4.0, 0.5 * (2**attempt))
            await asyncio.sleep(backoff)

    logger.warning(
        "%sLLM: Falha total apos %d tentativas -> Nenhuma decisao tomada",
        _cycle_prefix(log_cycle_id),
        total_tries,
    )
    return (None, False, "")
