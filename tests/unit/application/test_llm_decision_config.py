from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.application.services.llm import llm_decision as gem


def test_merge_generation_config_num_predict_caps():
    cfg = gem._merge_generation_config(0.0, 5000, {"top_p": 0.5}, system_instruction="s")
    assert cfg.max_output_tokens <= 4096
    assert pytest.approx(cfg.temperature) == 0.0


def test_merge_generation_config_extra_temperature_override():
    cfg = gem._merge_generation_config(
        0.5,
        None,
        {"temperature": 0.2, "max_output_tokens": 16},
        system_instruction="z",
    )
    assert pytest.approx(cfg.temperature) == 0.2
    assert cfg.max_output_tokens == 16


def test_merge_generation_config_top_k_and_candidate():
    cfg = gem._merge_generation_config(0.0, 10, {"top_k": 20, "candidate_count": 1}, system_instruction="z")
    assert cfg.top_k == 20
    assert cfg.candidate_count == 1


def test_merge_generation_config_thinking_budget_zero_e_safety_padrao():
    cfg = gem._merge_generation_config(0.0, 10, {}, system_instruction="z")
    assert cfg.thinking_config is None
    assert cfg.safety_settings is not None
    assert len(cfg.safety_settings) >= 4


def test_merge_generation_config_thinking_budget_conversion():
    extra = {"thinking_config": {"thinking_budget": 500}}
    cfg = gem._merge_generation_config(0.0, 10, extra, system_instruction="z")
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 500


def test_merge_generation_config_protected_keys_continue():
    extra = {"safety_settings": [], "system_instruction": "new", "top_p": 0.7}
    cfg = gem._merge_generation_config(0.0, 10, extra, system_instruction="fixed")
    assert cfg.system_instruction == "fixed"
    assert cfg.top_p == 0.7


def test_sync_generate_reads_response_text(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    resp = MagicMock()
    resp.text = " hi "
    fake_models = MagicMock()
    fake_models.generate_content.return_value = resp
    fake_client = MagicMock()
    fake_client.models = fake_models
    with patch("google.genai.Client", return_value=fake_client):
        cfg = gem._merge_generation_config(0.0, 8, {}, system_instruction="sys")
        out = gem._sync_generate("gemini-2.5-flash", "k", "prompt", cfg, 5.0)
    assert out == "hi"
    fake_models.generate_content.assert_called_once()


def test_sync_generate_corpo_vazio_emite_warning(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    buf: list[str] = []

    def cap(msg: str, *args: object) -> None:
        buf.append(msg % args if args else msg)

    monkeypatch.setattr(gem.logger, "warning", cap)
    resp = SimpleNamespace(text=None, candidates=[])
    fake_models = MagicMock()
    fake_models.generate_content.return_value = resp
    fake_client = MagicMock()
    fake_client.models = fake_models
    with patch("google.genai.Client", return_value=fake_client):
        cfg = gem._merge_generation_config(0.0, 8, {}, system_instruction="sys")
        out = gem._sync_generate("gemini-2.5-flash", "k", "prompt", cfg, 5.0)
    assert out == ""
    assert any("corpo vazio" in x for x in buf)
