import logging
from types import SimpleNamespace

from google.genai import types as genai_types

from src.application.services.llm import response_extract as gre


def test_visible_text_usa_agregado_sdk_quando_presente():
    r = SimpleNamespace(text="  PUT  ", candidates=[])
    assert gre.extract_llm_text(r) == "PUT"


def test_visible_text_concat_partes_quando_agregado_vazio():
    part = SimpleNamespace(text="CALL", thought=True)
    cand = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    r = SimpleNamespace(text=None, candidates=[cand])
    assert gre.extract_llm_text(r) == "CALL"


def test_visible_text_vazio_quando_partes_vazias():
    cand = SimpleNamespace(content=SimpleNamespace(parts=[]))
    r = SimpleNamespace(text=None, candidates=[cand])
    assert gre.extract_llm_text(r) == ""


def test_visible_text_vazio_quando_content_ausente():
    cand = SimpleNamespace(content=None)
    r = SimpleNamespace(text=None, candidates=[cand])
    assert gre.extract_llm_text(r) == ""


def test_log_llm_empty_response_diagnosticos_completos(monkeypatch):
    log = logging.getLogger("tgre")
    captured: list[str] = []

    def cap(msg: str, *args: object) -> None:
        captured.append(msg % args if args else msg)

    monkeypatch.setattr(log, "warning", cap)
    pf = SimpleNamespace(block_reason="OTHER")
    sr = SimpleNamespace(category="CAT", probability="HIGH")
    cand_a = SimpleNamespace(
        finish_reason="STOP",
        safety_ratings=[sr],
        content=SimpleNamespace(parts=[SimpleNamespace(text="")]),
    )
    gre.log_llm_empty_response(SimpleNamespace(prompt_feedback=pf, candidates=[cand_a]), log)
    cand_b = SimpleNamespace(finish_reason=None, safety_ratings=[], content=None)
    gre.log_llm_empty_response(SimpleNamespace(prompt_feedback=None, candidates=[cand_b]), log)
    blob = " ".join(captured)
    assert "prompt_block=" in blob
    assert "finish=" in blob
    assert "ratings=" in blob
    assert "text_parts_nonempty=0" in blob
    assert "content=None" in blob


def test_llm_default_safety_settings_lista_nao_vazia():
    ss = gre.llm_default_safety_settings(genai_types)
    assert len(ss) == 4
    assert all(getattr(x, "threshold", None) == genai_types.HarmBlockThreshold.BLOCK_ONLY_HIGH for x in ss)
