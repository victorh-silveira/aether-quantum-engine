from pathlib import Path
from unittest.mock import MagicMock, call, patch

from src.application.services.llm.llm_bridge_telemetry import (
    _truncate_preview,
    emit_llm_decision_log,
    emit_llm_http_snapshot,
)
from src.domain.models.trade import TradeDirection


def _base_kwargs(**overrides):
    defaults = {
        "cycle_id": 1,
        "logic_line_max_chars": 120,
        "direction": TradeDirection.CALL,
        "conviction": 0.8,
        "ref_px": 1.0,
        "model": "m",
        "mtf_alignment": "M30: alta",
        "justification": "x",
        "regime_label": "range",
        "atr_m5_pct": 0.01,
        "baseline_prob": 0.5,
        "wr_rolling": 0.5,
        "wr_samples": 10,
        "decision_source": "llm",
        "indicator_cfg": "cfg",
        "indicators_numeric_line": "line",
        "runtime_thresholds": "thresh",
        "prompt_char_count": 100,
        "prompt_audit_sections": [],
    }
    defaults.update(overrides)
    return defaults


def test_emit_llm_decision_log_emits_llm_audit_compact_when_audit_present():
    logger = MagicMock()
    audit = [
        ("MAPA_TF_CASCADE", "macro alta estrutura swing"),
        ("GATILHO_TF", "M1 rsi neutro"),
        ("ALINHAMENTO", "H1: alta | M5: baixa"),
        ("INDICADORES_MULTITF_LOG", "H1 indicadores (pre-calculados)"),
    ]
    emit_llm_decision_log(
        logger,
        "R_100",
        **_base_kwargs(
            cycle_id=5,
            logic_line_max_chars=100,
            conviction=0.7,
            mtf_alignment="x",
            justification="n",
            prompt_audit_sections=audit,
        ),
    )

    def _fmt(c):
        a = c.args
        return a[0] % tuple(a[1:]) if len(a) > 1 else str(a[0])

    debugs = [_fmt(c) for c in logger.debug.call_args_list if c.args]
    audit_lines = [x for x in debugs if "LLM_AUDIT" in x]
    assert len(audit_lines) == 2
    assert "line" in audit_lines[1]
    assert "[C0005]" in audit_lines[0]


def test_emit_llm_decision_log_inclui_wr_quando_rolling_definido():
    logger = MagicMock()
    emit_llm_decision_log(
        logger,
        "R_100",
        **_base_kwargs(
            wr_rolling=0.412,
            wr_samples=4,
            indicators_numeric_line="30:50/+0.1/A",
        ),
    )
    audit = [c for c in logger.debug.call_args_list if c.args and "LLM_AUDIT" in str(c.args[0])]
    assert "30:50/+0.1/A" in str(audit[1].args[0])


def test_emit_llm_decision_log_llm_dados_omite_placeholders_traco():
    logger = MagicMock()
    emit_llm_decision_log(
        logger,
        "R_100",
        **_base_kwargs(
            direction=None,
            ref_px=None,
            atr_m5_pct=None,
            baseline_prob=None,
            wr_rolling=None,
            wr_samples=0,
            indicators_numeric_line="",
        ),
    )
    dados = [c for c in logger.debug.call_args_list if c.args and "LLM_DADOS" in str(c.args[0])]
    msg = str(dados[0].args[0])
    assert "reg=range" in msg


def test_emit_llm_decision_log_emits_llm_perf_when_http_latency_positive():
    logger = MagicMock()
    emit_llm_decision_log(
        logger,
        "R_100",
        **_base_kwargs(
            cycle_id=2,
            conviction=0.85,
            justification="ok",
            llm_http_ms=123.4,
            llm_response_chars=500,
            engine_runtime={"num_predict": 768, "timeout": 120.0},
        ),
    )
    out = [c for c in logger.info.call_args_list if c.args and "LLM_RESPOSTA" in str(c.args[0])]
    assert out
    msg = str(out[0].args[0])
    assert "http_ms=123" in msg
    assert "policy=" not in msg


def test_emit_llm_decision_log_includes_entry_policy_tag():
    logger = MagicMock()
    emit_llm_decision_log(
        logger,
        "R_100",
        **_base_kwargs(
            cycle_id=3,
            direction=TradeDirection.PUT,
            conviction=0.9,
            justification="ok",
            entry_policy_tag="ENTRY_BLOCKED_BY_RSI_EXHAUSTION",
            engine_runtime={"num_predict": 768, "timeout": 120.0},
        ),
    )
    out = [c for c in logger.info.call_args_list if c.args and "LLM_RESPOSTA" in str(c.args[0])]
    assert out
    assert "PUT" in str(out[0].args[0])


def test_emit_llm_http_snapshot_leading_cycle_blank_before_llm_io():
    logger = MagicMock()
    emit_llm_http_snapshot(
        logger,
        "R_100",
        cycle_id=2,
        http_user="rsi=x",
        http_system="",
        sniper_tokens={},
        llm_config={"log_llm_io_line": True},
        leading_cycle_blank=True,
    )
    assert logger.info.call_args_list[0] == call("")


def test_emit_llm_http_snapshot_logs_io_and_writes_dump(tmp_path):
    logger = MagicMock()
    dump = tmp_path / "llm_http.json"
    emit_llm_http_snapshot(
        logger,
        "R_100",
        cycle_id=7,
        http_user="RSI=high, BB=inside",
        http_system="sys body",
        sniper_tokens={"rsi": "high", "bb": "inside"},
        llm_config={"log_llm_io_line": True, "log_llm_io_dump_path": str(dump)},
    )
    io_calls = [c for c in logger.info.call_args_list if c.args and "LLM_IO" in str(c.args[0])]
    assert len(io_calls) >= 2
    user_rendered = io_calls[0].args[0] % io_calls[0].args[1:]
    sys_rendered = io_calls[1].args[0] % io_calls[1].args[1:]
    assert "user_ch=19" in user_rendered
    assert "preview_user=RSI=high, BB=inside" in user_rendered
    assert "sys_ch=8" in sys_rendered
    assert "preview_sys=sys body" in sys_rendered
    text = dump.read_text(encoding="utf-8")
    assert '"cycle_id": 7' in text
    assert '"symbol": "R_100"' in text
    assert '"rsi": "high"' in text


def test_emit_llm_http_snapshot_skips_info_when_disabled(tmp_path):
    logger = MagicMock()
    emit_llm_http_snapshot(
        logger,
        "X",
        cycle_id=1,
        http_user="u",
        http_system="",
        sniper_tokens={},
        llm_config={"log_llm_io_line": False, "log_llm_io_dump_path": str(tmp_path / "n.json")},
    )
    assert not any(c.args and "LLM_IO" in str(c.args[0]) for c in logger.info.call_args_list)


def test_emit_llm_http_snapshot_relative_dump_resolves_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    logger = MagicMock()
    emit_llm_http_snapshot(
        logger,
        "S",
        cycle_id=9,
        http_user="x",
        http_system="",
        sniper_tokens={"rsi": "na"},
        llm_config={"log_llm_io_line": False, "log_llm_io_dump_path": "snap/rel.json"},
    )
    target = tmp_path / "snap" / "rel.json"
    assert target.is_file()
    assert '"cycle_id": 9' in target.read_text(encoding="utf-8")


def test_emit_llm_http_snapshot_preview_completo_sem_cap_preview(tmp_path):
    logger = MagicMock()
    long_u = "Z" * 500
    emit_llm_http_snapshot(
        logger,
        "R_100",
        cycle_id=3,
        http_user=long_u,
        http_system="",
        sniper_tokens={},
        llm_config={"log_llm_io_line": True},
    )
    io_calls = [c for c in logger.info.call_args_list if c.args and "LLM_IO" in str(c.args[0])]
    ic = io_calls[0]
    rendered = ic.args[0] % ic.args[1:] if len(ic.args) > 1 else str(ic.args[0])
    prev = rendered.split("preview_user=", 1)[1]
    assert prev == long_u


def test_emit_llm_http_snapshot_truncates_preview_when_long(tmp_path):
    logger = MagicMock()
    long_u = "Z" * 500
    emit_llm_http_snapshot(
        logger,
        "R_100",
        cycle_id=1,
        http_user=long_u,
        http_system="",
        sniper_tokens={},
        llm_config={"log_llm_io_line": True, "log_llm_io_preview_chars": 80},
    )
    io_calls = [c for c in logger.info.call_args_list if c.args and "LLM_IO" in str(c.args[0])]
    user_call = next(c for c in io_calls if "preview_user=" in (c.args[0] % c.args[1:]))
    rendered = user_call.args[0] % user_call.args[1:]
    assert "user_ch=500" in rendered
    prev = rendered.split("preview_user=", 1)[1]
    assert len(prev) == 80
    assert prev == "Z" * 80


def test_emit_llm_http_snapshot_jsonl_append(tmp_path):
    logger = MagicMock()
    dump = tmp_path / "llm_io.jsonl"
    emit_llm_http_snapshot(
        logger,
        "R_100",
        cycle_id=2,
        http_user="u",
        http_system="s",
        sniper_tokens={"hurst": "persist"},
        llm_config={"log_llm_io_line": False, "log_llm_io_dump_path": str(dump)},
        http_system_resolved="resolved",
        mtf_matrix="MTF_MATRIX: D1[...]",
    )
    lines = dump.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert '"mtf_matrix"' in lines[0]


def test_truncate_preview_short_under_cap():
    assert _truncate_preview("short", 80) == "short"


def test_emit_llm_http_snapshot_dump_fail_emits_warning(tmp_path):
    logger = MagicMock()
    dump = tmp_path / "llm_http.json"
    with patch.object(Path, "write_text", side_effect=OSError("x")):
        emit_llm_http_snapshot(
            logger,
            "S",
            cycle_id=2,
            http_user="u",
            http_system="",
            sniper_tokens={"rsi": "na"},
            llm_config={"log_llm_io_line": False, "log_llm_io_dump_path": str(dump)},
        )
    assert logger.warning.call_args_list
