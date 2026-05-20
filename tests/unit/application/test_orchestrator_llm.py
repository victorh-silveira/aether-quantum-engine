from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator import Orchestrator
from src.application.services.orchestrator.decision_mode_banner import emit_decision_engine_banner
from src.domain.models.trade import Contract, TradeDirection, TradeStatus


def test_emit_decision_engine_banner_llm_enabled(orch_config):
    orch_config["llm"] = {"enabled": True, "base_url": "http://127.0.0.1:11434", "model": "m"}
    log = MagicMock()
    emit_decision_engine_banner(log, orch_config, llm_enabled=True)
    assert log.debug.call_count >= 1
    joined = " ".join(str(c.args[0]) for c in log.debug.call_args_list if c.args)
    assert "modo=LLM" in joined


def test_emit_decision_engine_banner_simple_mode(orch_config):
    log = MagicMock()
    emit_decision_engine_banner(log, orch_config, llm_enabled=False)
    assert log.debug.call_count >= 1
    joined = " ".join(str(c.args[0]) for c in log.debug.call_args_list if c.args)
    assert "modo=simples" in joined


@pytest.mark.asyncio
async def test_trading_cycle_calls_llm_when_enabled(orch_config):
    orch_config["llm"] = {"enabled": True}
    fake_decisions = {
        "frxEURUSD": {"direction": TradeDirection.CALL, "metrics": {"conviction": 0.8}},
        "OTC_SPC": {"direction": TradeDirection.CALL, "metrics": {"conviction": 0.8}},
    }
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch.risk_manager.is_on_cooldown = MagicMock(return_value=False)
        orch.executor.execute_cluster = AsyncMock()
        with patch(
            "src.application.services.orchestrator.collect_llm_decisions",
            new_callable=AsyncMock,
            return_value=fake_decisions,
        ) as m_llm:
            await orch._run_trading_cycle_if_ready()
        m_llm.assert_awaited_once_with(orch)
        orch.executor.execute_cluster.assert_awaited_once_with(fake_decisions)


@pytest.mark.asyncio
async def test_trading_cycle_waits_when_has_pending_contracts(orch_config):
    orch_config["llm"] = {"enabled": True}
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()):
        orch = Orchestrator(orch_config, "token")
        orch.stream.is_synchronized = True
        orch.ws.is_running = True
        orch.risk_manager.is_on_cooldown = MagicMock(return_value=False)
        orch.executor.execute_cluster = AsyncMock()
        orch.state.active_contracts[1] = Contract(
            contract_id=1,
            proposal_id="p1",
            status=TradeStatus.OPEN,
            buy_price=10.0,
            payout=18.0,
            symbol="frxEURUSD",
            direction=TradeDirection.CALL,
            stake=10.0,
            expiry_time=0,
        )
        with patch(
            "src.application.services.orchestrator.collect_llm_decisions",
            new_callable=AsyncMock,
        ) as m_llm:
            await orch._run_trading_cycle_if_ready()
        m_llm.assert_not_awaited()
        orch.executor.execute_cluster.assert_not_awaited()
