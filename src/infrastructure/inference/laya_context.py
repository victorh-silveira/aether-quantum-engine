"""Orcamento exato e identidade do contexto entregue ao Laya System 1."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def serialize_laya_state(state: dict) -> str:
    """Serializa de forma identica em treinamento e inferencia, sem arredondamento extra."""
    return json.dumps(state, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def encode_laya_context(agent: Any, state: dict, questions: dict) -> tuple[str, int, str]:
    """Serializa todo o estado e reserva pergunta e separadores sem truncamento."""
    from laya.common import build_head, encode_text  # noqa: PLC0415

    text = serialize_laya_state(state)
    token_count = len(encode_text(agent.tok, text, add_special_tokens=False)["input_ids"])
    head_limit = int(agent.cfg.get("head_max_len", 192))
    head_sizes = []
    for question in questions.values():
        internal = {"t": question["type"], "ins": question["instructions"], "crit": question["criteria"]}
        head, _, _ = build_head(agent.tok, internal, head_limit)
        head_sizes.append(len(head))
    required = token_count + max(head_sizes) + 1
    capacity = int(agent.model.encoder.config.max_position_embeddings)
    if required > capacity:
        raise ValueError(f"LAYA_CONTEXT excede capacidade: tokens={required} capacidade={capacity}")
    signature = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return text, required, signature


def validate_laya_usage(raw: Any) -> dict:
    """Rejeita inferencia truncada ou sem comprovacao de consumo do contexto."""
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict) or "truncated" not in usage:
        raise ValueError("LAYA_CONTEXT resposta sem telemetria de truncamento")
    if usage["truncated"] or usage.get("state_tokens_dropped", 0):
        raise ValueError("LAYA_CONTEXT inferencia truncada; decisao invalida")
    return usage
