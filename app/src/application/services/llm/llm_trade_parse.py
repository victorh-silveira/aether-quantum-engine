"""Parse e validacao da resposta estruturada CALL/PUT da LLM."""

from __future__ import annotations

import json
import re
from typing import Any

from src.application.services.llm.llm_response_text import extract_think_block, preprocess_llm_response_text


LLM_TRADE_FORMAT_SUFFIX = (
    "\n\nSaida: objeto JSON com chaves EURUSD, US_CLUSTER, EU_CLUSTER, Probabilidade. Comece com {"
)

_DIR_TOKEN = r"(?:CALL|PUT|RISE|FALL|UP|DOWN)"
_FORMAT_LINE_RE = re.compile(
    rf"EURUSD\s*:\s*({_DIR_TOKEN})\s*\|\s*US(?:_CLUSTER)?\s*:\s*({_DIR_TOKEN})\s*\|\s*EU(?:_CLUSTER)?\s*:\s*({_DIR_TOKEN})"
    rf"\s*\|\s*PROBABILIDADE\s*:\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_LOOSE_DIR = re.compile(r"\b(CALL|PUT|RISE|FALL|UP|DOWN)\b", re.IGNORECASE)


def _norm_dir(token: str | None) -> str | None:
    """Normaliza token de direcao para CALL ou PUT."""
    val = str(token or "").strip().upper()
    if val in ("CALL", "PUT"):
        return val
    if val in ("RISE", "UP"):
        return "CALL"
    if val in ("FALL", "DOWN"):
        return "PUT"
    return None


def _conviction_from_value(raw: object) -> float:
    """Converte probabilidade bruta para conviccao entre 0.51 e 0.99."""
    try:
        val = float(raw)
    except TypeError:
        return 0.75
    except ValueError:
        return 0.75
    conv = val / 100.0 if val > 1.0 else val
    return max(0.51, min(0.99, conv))


def _strip_json_preamble(text: str) -> str:
    """Remove narrativa e cercas markdown antes do objeto JSON."""
    cleaned = preprocess_llm_response_text(text)
    if not cleaned:
        return ""
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return cleaned[start : end + 1]
    return cleaned.strip()


def _parse_json_trade(text: str) -> dict[str, Any] | None:
    """Tenta extrair objeto JSON com EURUSD, US_CLUSTER e EU_CLUSTER."""
    cleaned = _strip_json_preamble(text)
    if not cleaned:
        return None
    candidates = [cleaned]
    greedy = re.search(r"\{[\s\S]*\}", cleaned)
    if greedy:
        candidates.insert(0, greedy.group(0))
    for block in re.findall(r"\{[\s\S]*?\}", cleaned):
        candidates.append(block)
    for chunk in candidates:
        try:
            data = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        eurusd = _norm_dir(data.get("EURUSD") or data.get("eurusd") or data.get("direction") or data.get("EURUSD_dir"))
        us_c = _norm_dir(data.get("US_CLUSTER") or data.get("us_cluster") or data.get("US"))
        eu_c = _norm_dir(data.get("EU_CLUSTER") or data.get("eu_cluster") or data.get("EU"))
        prob_raw = data.get("Probabilidade") or data.get("probabilidade") or data.get("probability")
        if eurusd and us_c and eu_c:
            conv = _conviction_from_value(prob_raw if prob_raw is not None else 0.75)
            return {
                "direction": eurusd,
                "conviction": conv,
                "note": f"EURUSD_{eurusd}",
                "_direction_normalized": eurusd,
                "_conviction_normalized": conv,
                "us_cluster": us_c,
                "eu_cluster": eu_c,
            }
    return None


def _parse_format_line_trade(text: str) -> dict[str, Any] | None:
    """Extrai decisao quando a linha obrigatoria esta presente."""
    upper = preprocess_llm_response_text(text).upper()
    m = _FORMAT_LINE_RE.search(upper)
    if not m:
        return None
    eurusd = _norm_dir(m.group(1))
    us_c = _norm_dir(m.group(2))
    eu_c = _norm_dir(m.group(3))
    conv = _conviction_from_value(m.group(4))
    return {
        "direction": eurusd,
        "conviction": conv,
        "note": f"EURUSD_{eurusd}",
        "_direction_normalized": eurusd,
        "_conviction_normalized": conv,
        "us_cluster": us_c,
        "eu_cluster": eu_c,
    }


def _parse_loose_trade(text: str) -> dict[str, Any]:
    """Fallback regex para tags soltas na resposta."""
    raw = text or ""
    think, after_think = extract_think_block(raw)
    cleaned = preprocess_llm_response_text(after_think if think else raw)
    upper = cleaned.upper()

    m_prob = re.search(r"PROBABILIDADE\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", upper)
    m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%", cleaned)
    if m_prob:
        conv = _conviction_from_value(m_prob.group(1))
    elif m_pct:
        conv = _conviction_from_value(m_pct.group(1))
    else:
        conv = 0.75

    m_us = re.search(rf"US(?:_CLUSTER)?\s*[=:]\s*({_DIR_TOKEN})", upper)
    m_eu = re.search(rf"EU(?:_CLUSTER)?\s*[=:]\s*({_DIR_TOKEN})", upper)
    m_anchor = re.search(rf"EURUSD\s*:\s*({_DIR_TOKEN})", upper)

    direction = _norm_dir(m_anchor.group(1)) if m_anchor else None
    note = f"EURUSD_{direction}" if direction else "sniper_no_signal"
    if direction is None:
        upper_no_clusters = re.sub(rf"(US(?:_CLUSTER)?|EU(?:_CLUSTER)?)\s*[=:]\s*{_DIR_TOKEN}", "", upper)
        tokens = [_norm_dir(t) for t in _LOOSE_DIR.findall(upper_no_clusters)]
        tokens = [t for t in tokens if t]
        if len(tokens) == 1:
            direction = tokens[0]
            note = direction
        elif len(tokens) > 1:
            direction = tokens[0]
            note = f"{direction}_AMBIGUOUS"

    out: dict[str, Any] = {
        "direction": direction,
        "conviction": conv,
        "note": note,
        "_direction_normalized": direction,
        "_conviction_normalized": conv,
    }
    if m_us:
        out["us_cluster"] = _norm_dir(m_us.group(1))
    if m_eu:
        out["eu_cluster"] = _norm_dir(m_eu.group(1))
    if think:
        out["_think"] = think
    return out


def parse_llm_trade_response(text: str) -> dict[str, Any]:
    """Interpreta JSON, linha obrigatoria ou tags soltas CALL/PUT."""
    for parser in (_parse_json_trade, _parse_format_line_trade):
        parsed = parser(text)
        if parsed:
            return parsed
    return _parse_loose_trade(text)


def is_llm_trade_response_complete(parsed: dict[str, Any]) -> bool:
    """True quando EURUSD, US_CLUSTER e EU_CLUSTER estao CALL ou PUT."""
    return (
        parsed.get("direction") in ("CALL", "PUT")
        and parsed.get("us_cluster") in ("CALL", "PUT")
        and parsed.get("eu_cluster") in ("CALL", "PUT")
    )


def missing_llm_trade_fields(parsed: dict[str, Any]) -> list[str]:
    """Lista campos obrigatorios ausentes na resposta parseada."""
    missing: list[str] = []
    if parsed.get("direction") not in ("CALL", "PUT"):
        missing.append("EURUSD")
    if parsed.get("us_cluster") not in ("CALL", "PUT"):
        missing.append("US_CLUSTER")
    if parsed.get("eu_cluster") not in ("CALL", "PUT"):
        missing.append("EU_CLUSTER")
    return missing
