"""Unifica blocos JSON (gemini, llm_config, execution) sobre a config efetiva do motor."""

from __future__ import annotations

from contextlib import suppress
from copy import deepcopy
from typing import Any


def _maybe_set_max_predict(llm: dict[str, Any], raw: object) -> None:
    """Atribui max_predict_tokens quando o valor e um inteiro valido."""
    with suppress(TypeError, ValueError):
        llm["max_predict_tokens"] = int(raw)


def _merge_lc_temperature(llm: dict[str, Any], raw: object) -> None:
    """Mescla temperatura de llm_config em options quando o float e valido."""
    opt = dict(llm.get("options") or {})
    with suppress(TypeError, ValueError):
        opt["temperature"] = float(raw)
    llm["options"] = opt


def _merge_gemini_temperature(llm: dict[str, Any], raw: object) -> None:
    """Mescla temperatura do bloco gemini em generation_config."""
    gc = dict(llm.get("generation_config") or {})
    gc["temperature"] = 0.0
    with suppress(TypeError, ValueError):
        gc["temperature"] = float(raw)
    llm["generation_config"] = gc


def _restore_user_llm_over_gemini(llm: dict[str, Any], user_llm: dict[str, Any]) -> None:
    """Reaplica timeout, tokens, system_prompt e generation_config do bloco llm raiz apos gemini."""
    if user_llm.get("timeout_seconds") is not None:
        llm["timeout_seconds"] = float(user_llm["timeout_seconds"])
    if user_llm.get("llm_fallback_model") is not None:
        llm["llm_fallback_model"] = str(user_llm["llm_fallback_model"])
    if user_llm.get("max_predict_tokens") is not None:
        _maybe_set_max_predict(llm, user_llm["max_predict_tokens"])
    if user_llm.get("system_prompt") is not None:
        llm["system_prompt"] = str(user_llm["system_prompt"])
    ugc = user_llm.get("generation_config")
    if isinstance(ugc, dict) and ugc:
        merged = dict(llm.get("generation_config") or {})
        merged.update(ugc)
        llm["generation_config"] = merged


def _apply_gemini_fields(llm: dict[str, Any], gemini: dict[str, Any]) -> None:
    """Aplica campos do bloco gemini sobre o mapa llm."""
    if gemini.get("model") is not None:
        llm["model"] = str(gemini["model"])
    if gemini.get("llm_fallback_model") is not None:
        llm["llm_fallback_model"] = str(gemini["llm_fallback_model"])
    if gemini.get("timeout_seconds") is not None:
        llm["timeout_seconds"] = float(gemini["timeout_seconds"])
    if gemini.get("system_prompt") is not None:
        llm["system_prompt"] = str(gemini["system_prompt"])
    if gemini.get("num_predict") is not None:
        _maybe_set_max_predict(llm, gemini["num_predict"])
    if gemini.get("temperature") is not None:
        _merge_gemini_temperature(llm, gemini["temperature"])
    gc = gemini.get("generation_config")
    if isinstance(gc, dict):
        base = dict(llm.get("generation_config") or {})
        base.update(gc)
        llm["generation_config"] = base


def _apply_llm_config_fields(llm: dict[str, Any], lc: dict[str, Any]) -> None:
    """Aplica campos do bloco llm_config sobre o mapa llm efetivo."""
    if lc.get("model") is not None:
        llm["model"] = str(lc["model"])
    if lc.get("llm_fallback_model") is not None:
        llm["llm_fallback_model"] = str(lc["llm_fallback_model"])
    if lc.get("timeout_seconds") is not None:
        llm["timeout_seconds"] = float(lc["timeout_seconds"])
    if lc.get("keep_alive") is not None:
        llm["keep_alive"] = str(lc["keep_alive"])
    if lc.get("system_prompt") is not None:
        llm["system_prompt"] = str(lc["system_prompt"])
    if lc.get("num_predict") is not None:
        _maybe_set_max_predict(llm, lc["num_predict"])
    if lc.get("temperature") is not None:
        _merge_lc_temperature(llm, lc["temperature"])
    if lc.get("base_url") is not None:
        llm["base_url"] = str(lc["base_url"]).strip()
    lgc = lc.get("generation_config")
    if isinstance(lgc, dict):
        merged_gc = dict(llm.get("generation_config") or {})
        merged_gc.update(lgc)
        llm["generation_config"] = merged_gc


def effective_llm_section(root: dict[str, Any]) -> dict[str, Any]:
    """Mescla ``llm`` com ``gemini`` e ``llm_config`` sem alterar o dicionario raiz."""
    user_llm = deepcopy(root.get("llm") or {})
    llm = deepcopy(user_llm)
    gem = root.get("gemini")
    if isinstance(gem, dict):
        _apply_gemini_fields(llm, gem)
        _restore_user_llm_over_gemini(llm, user_llm)
    lc = root.get("llm_config")
    if isinstance(lc, dict):
        _apply_llm_config_fields(llm, lc)
    return llm


def merge_execution_section(root: dict[str, Any]) -> None:
    """Aplica ``execution`` sobre ``orchestrator`` quando declarado no JSON raiz."""
    block = root.get("execution")
    if not isinstance(block, dict):
        return
    orch = root.setdefault("orchestrator", {})
    inner = orch.setdefault("execution", {})
    nested = {
        "include_anchor_trades",
        "inter_symbol_delay",
        "settlement_max_stagnant_polls",
        "settlement_poll_seconds",
        "settlement_request_timeout_seconds",
    }
    for key, val in block.items():
        if key in nested:
            inner[key] = val
        elif key != "execution":
            orch[key] = val
    sub = block.get("execution")
    if isinstance(sub, dict):
        for key, val in sub.items():
            inner[key] = val


def risk_limits_section(root: dict[str, Any]) -> dict[str, Any]:
    """Retorna o mapa ``risk_management.limits`` normalizado."""
    rm = root.get("risk_management")
    if not isinstance(rm, dict):
        return {}
    lim = rm.get("limits")
    return lim if isinstance(lim, dict) else {}
