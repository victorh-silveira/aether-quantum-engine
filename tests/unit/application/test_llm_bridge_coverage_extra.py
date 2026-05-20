from unittest.mock import MagicMock

import pytest

from src.application.services.llm.llm_bridge_telemetry import emit_llm_decision_log
from src.application.services.llm.llm_bridge_utils import parse_llm_trade_response
from src.application.services.llm.symbol_decision_utils import _fetch_cluster_status
from src.domain.models.trade import TradeDirection


def test_llm_bridge_utils_coverage_put():
    out = parse_llm_trade_response("EURUSD: WAIT")
    assert out["direction"] is None
    assert out["note"] == "EURUSD_WAIT"


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
async def test_fetch_cluster_status_exception_coverage():
    class DummyStream:
        def fetch_candle_closes(self, *_a, **_k):
            raise RuntimeError("simulated error")

        fetch_candle_closes._is_coroutine = True

    class DummyOrch:
        stream = DummyStream()
        logger = MagicMock()

    orch = DummyOrch()
    # To force the exception inside the try block, we make the method raise directly.
    # Actually wait, if fetch_candle_closes is just a sync function that raises RuntimeError,
    # the list comprehension *[orch.stream.fetch_candle_closes...] will raise RuntimeError synchronously!
    # And that is caught by `except Exception as e:` in the function!
    res = await _fetch_cluster_status(orch, {})
    assert res == ""
    orch.logger.warning.assert_called()
