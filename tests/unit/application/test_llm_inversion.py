from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.global_macro_confluence import empty_macro_snapshot
from src.application.services.llm.symbol_decision import collect_symbol_llm_decision
from src.domain.models.trade import TradeDirection


def _prompt_return(ctx):
    return ("prompt", ctx, {}, 0.5, "line", "bundle", "m", "st", "sw", "tr", "mtf", [], empty_macro_snapshot())


def mock_llm_metrics(direction, conviction, note):
    return {
        "direction": direction.name if direction else "NONE",
        "conviction": conviction,
        "llm_note": note,
        "execute": direction is not None,
    }


@pytest.mark.asyncio
async def test_collect_symbol_llm_decision_inverts_low_conviction():
    """Valida que a inversao ocorre quando a conviccao esta abaixo do threshold de inversao."""
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.symbols = ["frxEURUSD"]
    orch.logger = MagicMock()

    ctx = {"hurst_value": 0.6}
    with (
        patch(
            "src.application.services.llm.symbol_decision.build_symbol_prompt",
            AsyncMock(return_value=_prompt_return(ctx)),
        ),
        patch(
            "src.application.services.llm.symbol_decision._request_payload",
            AsyncMock(
                return_value={"_direction_normalized": "CALL", "_conviction_normalized": 0.55, "note": "original_note"}
            ),
        ),
    ):
        runtime = {
            "inversion_threshold": 0.65,
            "follow_threshold": 0.85,
            "indicator_config": IndicatorConfig(),
            "payout_estimate": 0.85,
            "min_payout_accept": 0.80,
            "duration": 5,
            "du": "m",
            "model": "m",
            "base_url": "http://x",
            "timeout": 10,
            "num_predict": 64,
            "min_conviction_execute": 0.51,
        }

        direction, metrics = await collect_symbol_llm_decision(
            orch, sym="frxEURUSD", runtime=runtime, llm_metrics=mock_llm_metrics
        )

        assert direction == TradeDirection.PUT
        assert metrics["llm_exec_inverted"] is True


@pytest.mark.asyncio
async def test_collect_symbol_llm_decision_follows_noise_zone():
    """Valida que o bot SEGUE trades mesmo na zona de ruido (Sempre Operar)."""
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.symbols = ["frxEURUSD"]
    orch.logger = MagicMock()

    ctx = {"hurst_value": 0.6}
    with (
        patch(
            "src.application.services.llm.symbol_decision.build_symbol_prompt",
            AsyncMock(return_value=_prompt_return(ctx)),
        ),
        patch(
            "src.application.services.llm.symbol_decision._request_payload",
            AsyncMock(
                return_value={"_direction_normalized": "CALL", "_conviction_normalized": 0.75, "note": "original_note"}
            ),
        ),
    ):
        runtime = {
            "inversion_threshold": 0.65,
            "follow_threshold": 0.85,
            "indicator_config": IndicatorConfig(),
            "payout_estimate": 0.85,
            "min_payout_accept": 0.80,
            "duration": 5,
            "du": "m",
            "model": "m",
            "base_url": "http://x",
            "timeout": 10,
            "num_predict": 64,
            "min_conviction_execute": 0.51,
        }

        direction, metrics = await collect_symbol_llm_decision(
            orch, sym="frxEURUSD", runtime=runtime, llm_metrics=mock_llm_metrics
        )

        assert direction == TradeDirection.CALL
        assert "Follow (Noise Zone)" in metrics["llm_note"]


@pytest.mark.asyncio
async def test_collect_symbol_llm_decision_follows_high_conviction():
    """Valida que o sinal e seguido quando a conviccao esta acima do threshold de follow."""
    orch = MagicMock()
    orch._active_cycle_id = 1
    orch.symbols = ["frxEURUSD"]
    orch.logger = MagicMock()

    ctx = {"hurst_value": 0.6}
    with (
        patch(
            "src.application.services.llm.symbol_decision.build_symbol_prompt",
            AsyncMock(return_value=_prompt_return(ctx)),
        ),
        patch(
            "src.application.services.llm.symbol_decision._request_payload",
            AsyncMock(
                return_value={"_direction_normalized": "CALL", "_conviction_normalized": 0.90, "note": "original_note"}
            ),
        ),
    ):
        runtime = {
            "inversion_threshold": 0.65,
            "follow_threshold": 0.85,
            "indicator_config": IndicatorConfig(),
            "payout_estimate": 0.85,
            "min_payout_accept": 0.80,
            "duration": 5,
            "du": "m",
            "model": "m",
            "base_url": "http://x",
            "timeout": 10,
            "num_predict": 64,
            "min_conviction_execute": 0.51,
        }

        direction, metrics = await collect_symbol_llm_decision(
            orch, sym="frxEURUSD", runtime=runtime, llm_metrics=mock_llm_metrics
        )

        assert direction == TradeDirection.CALL
        assert metrics["llm_exec_inverted"] is False
