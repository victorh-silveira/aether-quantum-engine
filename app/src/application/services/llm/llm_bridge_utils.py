"""Utilitarios de processamento e parsing para o bridge LLM."""

from __future__ import annotations

import re

from src.application.services.llm.llm_response_text import extract_think_block, preprocess_llm_response_text
from src.application.services.llm.llm_trade_parse import (
    is_llm_trade_response_complete,
    missing_llm_trade_fields,
    parse_llm_trade_response,
)


__all__ = [
    "canonical_direction_token",
    "extract_think_block",
    "is_llm_trade_response_complete",
    "missing_llm_trade_fields",
    "parse_llm_trade_response",
    "preprocess_llm_response_text",
    "score_token",
    "strict_normalize_direction",
    "trend_token",
]


def canonical_direction_token(raw: str) -> str | None:
    """Normaliza um token para CALL, PUT ou None."""
    if not raw:
        return None
    upper = str(raw).strip().upper()
    if "CALL" in upper:
        return "CALL"
    if "PUT" in upper:
        return "PUT"
    if any(x in upper for x in ("WAIT", "SKIP", "HOLD")):
        return None
    return None


def strict_normalize_direction(raw: str) -> str | None:
    """Extrai um unico token CALL ou PUT; ambiguidade retorna None."""
    cleaned = preprocess_llm_response_text(raw or "").strip()
    if not cleaned:
        return None
    upper = cleaned.upper()
    found = re.findall(r"\b(CALL|PUT)\b", upper)
    if not found:
        return None
    uniq: list[str] = []
    for tok in found:
        if tok not in uniq:
            uniq.append(tok)
    if len(uniq) > 1:
        return None
    return uniq[0]


def trend_token(text: str) -> str | None:
    """Normaliza texto de tendencia para token alto/baixo."""
    lowered = (text or "").lower()
    if any(x in lowered for x in ("alta", "alto", "bull", "bullish", "up", "compra", "comprador")):
        return "alta"
    if any(x in lowered for x in ("baixa", "baixo", "bear", "bearish", "down", "venda", "vendedor")):
        return "baixa"
    return None


def score_token(text: str) -> int:
    """Converte texto de tendencia em score discreto."""
    tk = trend_token(text)
    if tk == "alta":
        return 1
    if tk == "baixa":
        return -1
    return 0
