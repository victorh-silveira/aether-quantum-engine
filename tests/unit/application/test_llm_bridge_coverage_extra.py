from unittest.mock import MagicMock

import pytest

from src.application.services.llm.llm_bridge_telemetry import emit_llm_decision_log
from src.application.services.llm.llm_bridge_utils import parse_llm_trade_response
from src.application.services.llm.symbol_decision_utils import fetch_macro_snapshot
from src.domain.models.trade import TradeDirection


def test_llm_bridge_utils_coverage_put():
    out = parse_llm_trade_response("EURUSD: WAIT")
    assert out["direction"] is None
    assert out["note"] == "sniper_no_signal"


def test_llm_bridge_telemetry_cluster_coverage():
    logger = MagicMock()
    emit_llm_decision_log(
        logger,
        "frxEURUSD",
        cycle_id=1,
        logic_line_max_chars=10,
        direction=TradeDirection.CALL,
        conviction=0.9,
        ref_px=1.0,
        model="x",
        mtf_alignment="",
        justification="",
        regime_label="",
        atr_m5_pct=0.0,
        baseline_prob=0.0,
        wr_rolling=0.0,
        wr_samples=0,
        decision_source="",
        indicator_cfg="",
        indicators_numeric_line="",
        runtime_thresholds="",
        prompt_char_count=0,
        prompt_audit_sections=[],
        us_cluster="CALL",
        eu_cluster="PUT",
    )


@pytest.mark.asyncio
async def test_fetch_macro_snapshot_exception_coverage():
    class DummyStream:
        def fetch_candle_closes(self, *_a, **_k):
            raise RuntimeError("simulated error")

        fetch_candle_closes._is_coroutine = True

    class DummyOrch:
        stream = DummyStream()
        logger = MagicMock()
        config = {"strategy": {"clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FCHI"]}}}

    orch = DummyOrch()
    snap = await fetch_macro_snapshot(orch, {})
    assert snap.tag == "indefinido"
    assert snap.cluster_status == ""
    orch.logger.warning.assert_called()
