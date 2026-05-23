"""Extracao de texto util e diagnostico para respostas generate_content (google-genai)."""

from __future__ import annotations

import json
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


def _part_is_thought(part: Any) -> bool:
    """True quando a parte da resposta e conteudo interno de raciocinio."""
    return bool(getattr(part, "thought", False))


def _json_blob_from_parsed(obj: Any) -> str:
    """Serializa objeto parsed da API em string JSON ou retorna vazio."""
    if obj is None:
        return ""
    if isinstance(obj, dict):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, str) and obj.strip().startswith("{"):
        return obj.strip()
    return ""


def extract_llm_text(resp: Any) -> str:
    """Le texto agregado ou concatena partes visiveis (sem thought)."""
    parsed = getattr(resp, "parsed", None)
    blob = _json_blob_from_parsed(parsed)
    if blob:
        return blob
    raw = getattr(resp, "text", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    candidates = getattr(resp, "candidates", None) or []
    if not candidates:
        return ""
    c0 = candidates[0]
    cand_parsed = _json_blob_from_parsed(getattr(c0, "parsed", None))
    if cand_parsed:
        return cand_parsed
    content = getattr(c0, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    pieces: list[str] = []
    for part in parts:
        if _part_is_thought(part):
            continue
        txt = getattr(part, "text", None)
        if isinstance(txt, str) and txt.strip():
            pieces.append(txt)
    return "".join(pieces).strip()


def response_finish_reason(resp: Any) -> str:
    """Retorna finish_reason do primeiro candidato ou string vazia."""
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return ""
    fr = getattr(cands[0], "finish_reason", None)
    return str(fr or "")


def is_max_tokens_finish(resp: Any) -> bool:
    """Indica truncamento por limite de tokens na resposta."""
    fr = response_finish_reason(resp).upper()
    return "MAX_TOKENS" in fr or "MAX_OUTPUT" in fr


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
        thought_n = sum(1 for p in prts if _part_is_thought(p))
        nonempty = sum(
            1
            for p in prts
            if not _part_is_thought(p)
            and isinstance(getattr(p, "text", None), str)
            and str(getattr(p, "text", "")).strip()
        )
        bits.append(f"parts={len(prts)} thought_parts={thought_n} text_parts_nonempty={nonempty}")
    log.warning("LLM Gemini corpo vazio: %s", " ".join(bits))
