from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.llm_bridge import llm_metrics
from src.application.services.llm.symbol_decision import collect_symbol_llm_decision
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_collect_symbol_llm_decision_failsafe_force_direction():
    """Valida o fail-safe que força uma direção quando a LLM retorna None."""
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.symbols = ["1HZ10V"]
    orch.logger = MagicMock()

    ctx = {"zscore_value": 2.5, "regime_label": "range"}

    with (
        patch(
            "src.application.services.llm.symbol_decision.build_symbol_prompt",
            AsyncMock(return_value=("prompt", ctx, {}, 0.5, "line", "bundle", "m", "st", "sw", "tr", "mtf", [])),
        ),
        patch(
            "src.application.services.llm.symbol_decision._request_payload",
            AsyncMock(return_value={"_direction_normalized": None, "_conviction_normalized": 0.0, "note": "error"}),
        ),
    ):
        runtime = {
            "inversion_threshold": 0.30,
            "follow_threshold": 0.40,
            "indicator_config": IndicatorConfig(),
            "payout_estimate": 0.85,
            "min_payout_accept": 0.80,
            "duration": 5,
            "du": "m",
            "model": "m",
            "base_url": "http://x",
            "timeout": 10,
            "num_predict": 64,
            "min_conviction_execute": 0.55,
        }

        direction, metrics = await collect_symbol_llm_decision(
            orch, sym="1HZ10V", runtime=runtime, llm_metrics=llm_metrics
        )

        assert direction == TradeDirection.PUT
        assert "FORCED EXEC" in metrics["llm_note"]
        assert metrics["conviction"] == 0.56
        assert metrics["execute"] is True
