"""Regressoes do orcamento de contexto e da resposta de inferencia Laya."""

import json
from types import SimpleNamespace

import pytest

from src.infrastructure.inference.laya_context import encode_laya_context, validate_laya_usage
from src.infrastructure.inference.laya_decision_adapter import _LAYA_QUESTIONS, _extract_choice_probs


class CharacterTokenizer:
    """Tokenizer deterministico para conferir orcamentos e limites sem pesos."""

    mask_token = "<mask>"
    mask_token_id = 1
    cls_token_id = 2
    sep_token_id = 3

    def __call__(self, text, **kwargs):
        """Mapeia caracteres e respeita o limite explicito das opcoes."""
        ids = [ord(c) for c in text]
        return {"input_ids": ids[: kwargs["max_length"]] if kwargs.get("truncation") else ids}


def context_agent(capacity=8192):
    """Fornece a interface do encoder real usada pelo orcamento."""
    return SimpleNamespace(
        tok=CharacterTokenizer(),
        cfg={"head_max_len": 192},
        model=SimpleNamespace(encoder=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=capacity))),
    )


def test_context_budget_preserves_full_state_and_changes_signature():
    state = {"indicators": {"ema_20": 1.0}, "recent_candles": [{"epoch": i} for i in range(12)]}
    text, required, signature = encode_laya_context(context_agent(), state, _LAYA_QUESTIONS)
    assert json.loads(text) == state
    assert required > len(text)
    assert encode_laya_context(context_agent(required), state, _LAYA_QUESTIONS)[1] == required
    with pytest.raises(ValueError, match="excede capacidade"):
        encode_laya_context(context_agent(required - 1), state, _LAYA_QUESTIONS)
    state["indicators"]["ema_20"] = 2.0
    assert encode_laya_context(context_agent(), state, _LAYA_QUESTIONS)[2] != signature


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {},
        {"usage": {}},
        {"usage": {"truncated": True}},
        {"usage": {"truncated": False, "state_tokens_dropped": 1}},
    ],
)
def test_invalid_usage_never_becomes_valid_decision(raw):
    with pytest.raises(ValueError, match="LAYA_CONTEXT"):
        validate_laya_usage(raw)


def test_usage_and_zero_probabilities(caplog):
    assert validate_laya_usage({"usage": {"truncated": False}}) == {"truncated": False}
    _, p_call, p_put, conf = _extract_choice_probs(
        {"answers": {"direction": {"probabilities": {"CALL": 0.0, "PUT": 1.0}, "confidence": "bad"}}}, {}
    )
    assert (p_call, p_put, conf) == (0.0, 1.0, 1.0)
    assert "fallback" in caplog.text
    assert _extract_choice_probs({"probabilities": {"direction": {"CALL": 0, "PUT": 0}}}, {})[1:3] == (0.5, 0.5)
    assert (
        _extract_choice_probs(
            {"probabilities": {"direction": {"CALL": 0.5000000001, "PUT": 0.5}}}, {"ema_20": 0, "ema_50": 1}
        )[0].name
        == "CALL"
    )
