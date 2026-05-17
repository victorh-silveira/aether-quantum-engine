from unittest.mock import MagicMock

from src.application.services.llm.llm_bridge_telemetry import emit_llm_decision_log


def test_emit_llm_decision_log_skip_branch():
    logger = MagicMock()

    emit_llm_decision_log(
        logger,
        "R_100",
        cycle_id=1,
        logic_line_max_chars=100,
        direction=None,
        conviction=0.0,
        ref_px=100.0,
        model="m",
        mtf_alignment="T/T/T",
        justification="SKIP",
        regime_label="trend",
        atr_m5_pct=0.01,
        baseline_prob=0.5,
        wr_rolling=0.5,
        wr_samples=10,
        decision_source="llm_skip",
        indicator_cfg="cfg",
        indicators_numeric_line="num",
        runtime_thresholds="thr",
        prompt_char_count=100,
        prompt_audit_sections=[],
    )

    calls = [str(c) for c in logger.info.call_args_list]
    assert any("[SKIP]" in c for c in calls)
