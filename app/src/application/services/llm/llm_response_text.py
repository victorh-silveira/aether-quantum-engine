"""Limpeza e extracao de texto bruto da resposta LLM."""

from __future__ import annotations

import re


def preprocess_llm_response_text(text: str) -> str:
    """Limpa blocos de codigo markdown e caracteres de controle do texto da LLM."""
    if not text:
        return ""
    text = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", text)
    text = re.sub(r"\*{1,3}", "", text)
    return text.strip()


def extract_think_block(raw_text: str) -> tuple[str, str]:
    """Separa bloco think e texto estruturado final da resposta."""
    text = str(raw_text or "")
    m = re.search(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
    if not m:
        return "", text.strip()
    think = " ".join(m.group(1).split()).strip()
    structured = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    return think, structured
