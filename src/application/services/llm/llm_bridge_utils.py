"""Utilitarios de processamento e parsing para o bridge LLM."""

from __future__ import annotations

import re
from typing import Any


def preprocess_llm_response_text(text: str) -> str:
    """Limpa blocos de codigo markdown e caracteres de controle do texto da LLM."""
    if not text:
        return ""
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text)
    text = re.sub(r"\*{1,3}", "", text)
    return text.strip()


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
    """Extrai um unico token CALL, PUT ou WAIT; ambiguidade vira WAIT."""
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


def extract_think_block(raw_text: str) -> tuple[str, str]:
    """Separa bloco think e texto estruturado final da resposta."""
    text = str(raw_text or "")
    m = re.search(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
    if not m:
        return "", text.strip()
    think = " ".join(m.group(1).split()).strip()
    structured = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    return think, structured


def parse_llm_trade_response(text: str) -> dict[str, Any]:
    """Interpreta texto bruto: extrai direcao, clusters e probabilidade %."""
    raw = text or ""
    think, after_think = extract_think_block(raw)
    cleaned = preprocess_llm_response_text(after_think if think else raw)
    upper = cleaned.upper()

    m_pct = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if m_pct:
        val = float(m_pct.group(1))
        conv = val / 100.0 if val > 1.0 else val
    else:
        conv = 1.0
    conv = max(0.51, min(0.99, conv))

    m_us = re.search(r"US_CLUSTER:\s*(CALL|PUT)", upper)
    m_eu = re.search(r"EU_CLUSTER:\s*(CALL|PUT)", upper)
    m_anchor = re.search(r"EURUSD:\s*([A-Z]+)", upper)

    # Remove clusters from the string before doing a global search for CALL/PUT to avoid leaking cluster direction
    upper_no_clusters = re.sub(r"(US_CLUSTER|EU_CLUSTER):\s*[A-Z]+", "", upper)
    has_call = "CALL" in upper_no_clusters
    has_put = "PUT" in upper_no_clusters

    if m_anchor:
        raw_dir = m_anchor.group(1)
        if raw_dir in ("CALL", "PUT"):
            direction = raw_dir
            note = f"EURUSD_{direction}"
        else:
            direction = None
            note = f"EURUSD_{raw_dir}"
    elif has_call and has_put:
        first_call = upper_no_clusters.find("CALL")
        first_put = upper_no_clusters.find("PUT")
        direction = "CALL" if first_call < first_put else "PUT"
        note = f"{direction}_AMBIGUOUS"
    elif has_call:
        direction = "CALL"
        note = "CALL"
    elif has_put:
        direction = "PUT"
        note = "PUT"
    else:
        direction = None
        note = "sniper_no_signal"

    out: dict[str, Any] = {
        "direction": direction,
        "conviction": conv,
        "note": note,
        "_direction_normalized": direction,
        "_conviction_normalized": conv,
    }
    if m_us:
        out["us_cluster"] = m_us.group(1)
    if m_eu:
        out["eu_cluster"] = m_eu.group(1)

    if think:
        out["_think"] = think
    return out


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
