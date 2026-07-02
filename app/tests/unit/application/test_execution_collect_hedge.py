from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.execution_symbols import (
    has_recovery_hedge_candidate,
    inject_recovery_hedge_candidates,
)
from src.application.services.orchestrator.execution_collect import apply_recovery_hedge_to_candidates
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, HEDGE_PEER_SYMBOL, PAIR


def test_inject_recovery_hedge_early_returns():
    base = [(PAIR, TradeDirection.CALL, {"execute": True})]
    assert (
        inject_recovery_hedge_candidates(
            base,
            {},
            last_loss_symbol=None,
            last_loss_direction=None,
        )
        == base
    )
    present = [(HEDGE_PEER_SYMBOL, TradeDirection.PUT, {"execute": True})]
    assert (
        inject_recovery_hedge_candidates(
            present,
            {HEDGE_PEER_SYMBOL: {"direction": TradeDirection.CALL, "metrics": {}}},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        )
        == present
    )
    assert (
        inject_recovery_hedge_candidates(
            base,
            {HEDGE_PEER_SYMBOL: {"direction": None, "metrics": {}}},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        )
        == base
    )
    assert has_recovery_hedge_candidate(base, last_loss_symbol=None, last_loss_direction=None)


def test_apply_recovery_hedge_keeps_same_direction_candidates():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        ),
        _active_cycle_id=2,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(PAIR, TradeDirection.CALL, {"execute": True})]
    result = apply_recovery_hedge_to_candidates(
        exec_mgr,
        candidates,
        {},
        cid="C0002",
    )
    assert result == candidates


def test_apply_recovery_hedge_keeps_candidates_for_market_ranking():
    orch = SimpleNamespace(
        config={
            "risk_management": {"kelly": {}},
            "orchestrator": {"execution": {}},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 5.0},
            last_loss_symbol=PAIR,
            last_loss_direction="PUT",
        ),
        _active_cycle_id=3,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _trade_symbols=lambda: [ANCHOR, PAIR],
    )
    candidates = [(ANCHOR, TradeDirection.CALL, {"execute": True})]
    result = apply_recovery_hedge_to_candidates(
        exec_mgr,
        candidates,
        {},
        cid="C0003",
        mandatory=True,
    )
    assert result == candidates


def test_apply_recovery_hedge_passthrough_without_pending():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        risk_manager=SimpleNamespace(pending_loss={}, last_loss_symbol=None, last_loss_direction=None),
        _active_cycle_id=1,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(ANCHOR, TradeDirection.CALL, {})]
    assert apply_recovery_hedge_to_candidates(exec_mgr, candidates, {}, cid="C0001") == candidates


def test_apply_recovery_hedge_keeps_pool_when_direction_differs_from_loss():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="PUT",
        ),
        _active_cycle_id=7,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(ANCHOR, TradeDirection.CALL, {"execute": True})]
    result = apply_recovery_hedge_to_candidates(
        exec_mgr,
        candidates,
        {},
        cid="C0007",
    )
    assert result == candidates
