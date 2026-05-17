"""Extracao de texto util e diagnostico para respostas generate_content (google-genai)."""

from __future__ import annotations

import logging
from typing import Any


def llm_default_safety_settings(types_mod: Any) -> list[Any]:
    """Define bloqueios apenas para probabilidade alta, reduzindo respostas vazias por filtro."""
    t = types_mod.HarmBlockThreshold.BLOCK_ONLY_HIGH
    cats = (
        types_mod.HarmCategory.HARM_CATEGORY_HARASSMENT,
        types_mod.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types_mod.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types_mod.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    )
    return [types_mod.SafetySetting(category=c, threshold=t) for c in cats]


def extract_llm_text(resp: Any) -> str:
    """Le texto agregado ou concatena partes, inclusive onde o SDK omite thought no agregado."""
    raw = getattr(resp, "text", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return ""
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    pieces: list[str] = []
    for part in parts:
        txt = getattr(part, "text", None)
        if isinstance(txt, str) and txt.strip():
            pieces.append(txt)
    return "".join(pieces).strip()


def log_llm_empty_response(resp: Any, log: logging.Logger) -> None:
    """Emite aviso com motivo provavel quando nao ha texto util na resposta."""
    bits: list[str] = []
    pf = getattr(resp, "prompt_feedback", None)
    if pf is not None:
        br = getattr(pf, "block_reason", None)
        if br is not None:
            bits.append(f"prompt_block={br}")
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        log.warning("LLM Gemini corpo vazio: candidates=0 %s", " ".join(bits))
        return
    c0 = cands[0]
    fr = getattr(c0, "finish_reason", None)
    if fr is not None:
        bits.append(f"finish={fr}")
    srs = getattr(c0, "safety_ratings", None) or []
    if srs:
        pairs: list[str] = []
        for r in srs[:8]:
            cat = getattr(r, "category", None)
            pr = getattr(r, "probability", None)
            pairs.append(f"{cat}:{pr}")
        bits.append("ratings=" + ";".join(pairs))
    content = getattr(c0, "content", None)
    if content is None:
        bits.append("content=None")
    else:
        prts = getattr(content, "parts", None) or []
        nonempty = sum(
            1 for p in prts if isinstance(getattr(p, "text", None), str) and str(getattr(p, "text", "")).strip()
        )
        bits.append(f"parts={len(prts)} text_parts_nonempty={nonempty}")
    log.warning("LLM Gemini corpo vazio: %s", " ".join(bits))
