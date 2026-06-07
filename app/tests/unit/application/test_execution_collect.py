from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.execution_symbols import (
    has_recovery_hedge_candidate,
    inject_recovery_hedge_candidates,
    select_mandatory_execution_candidate,
)
from src.application.services.orchestrator.execution_collect import (
    apply_recovery_hedge_to_candidates,
    collect_cluster_orders,
)
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
    present = [(HEDGE_PEER_SYMBOL, TradeDirection.CALL, {"execute": True})]
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


def test_apply_recovery_hedge_returns_expanded_candidates():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"recovery_require_hedge": True}}},
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
        {HEDGE_PEER_SYMBOL: {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.6}}},
        cid="C0002",
    )
    assert len(result) == 2
    assert any(item[0] == HEDGE_PEER_SYMBOL and item[1] == TradeDirection.CALL for item in result)


def test_apply_recovery_hedge_passthrough_without_pending():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"recovery_require_hedge": True}}},
        risk_manager=SimpleNamespace(pending_loss={}, last_loss_symbol=None, last_loss_direction=None),
        _active_cycle_id=1,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(ANCHOR, TradeDirection.CALL, {})]
    assert apply_recovery_hedge_to_candidates(exec_mgr, candidates, {}, cid="C0001") == candidates


def test_collect_cluster_orders_empty_after_recovery_skip():
    orch = SimpleNamespace(
        anchor=ANCHOR,
        symbols=[ANCHOR, PAIR],
        config={
            "orchestrator": {"execution": {"include_anchor_trades": False, "recovery_require_hedge": True}},
            "deep_learning": {"post_loss_flip_raw_min": 0.62},
        },
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        ),
        _active_cycle_id=9,
    )
    exec_mgr = SimpleNamespace(
        orch=orch,
        logger=MagicMock(),
        _execution_flags=lambda: (True, False),
        _trade_symbols=lambda: [PAIR],
    )
    decisions = {
        PAIR: {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.4, "execute": True}},
    }
    assert collect_cluster_orders(exec_mgr, decisions) == []


def test_apply_recovery_hedge_skip_when_require_hedge_and_missing():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"recovery_require_hedge": True}}},
        risk_manager=SimpleNamespace(
            pending_loss={PAIR: 10.0},
            last_loss_symbol=PAIR,
            last_loss_direction="CALL",
        ),
        _active_cycle_id=7,
    )
    exec_mgr = SimpleNamespace(orch=orch, logger=MagicMock())
    candidates = [(ANCHOR, TradeDirection.CALL, {"execute": True})]
    result = apply_recovery_hedge_to_candidates(
        exec_mgr,
        candidates,
        {PAIR: {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.6}}},
        cid="C0007",
    )
    assert result == []
    exec_mgr.logger.warning.assert_called_once()


def test_select_mandatory_empty_pool_fallback():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.99}})
    candidates = [
        (ANCHOR, TradeDirection.CALL, {"trade_score": 0.70, "execute": False}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.40, "execute": False}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        diversify_margin=0.08,
        recovery_active=True,
        flip_raw_min=0.99,
    )
    assert best[0] == ANCHOR


def test_select_mandatory_recovery_uses_hedge_pool():
    orch = SimpleNamespace(config={"deep_learning": {"post_loss_flip_raw_min": 0.99}})
    candidates = [
        (HEDGE_PEER_SYMBOL, TradeDirection.PUT, {"trade_score": 0.55, "val_accuracy": 0.60, "execute": True}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.40, "execute": False}),
    ]
    best = select_mandatory_execution_candidate(
        orch,
        candidates,
        last_loss_symbol=PAIR,
        last_loss_direction="CALL",
        diversify_margin=0.08,
        recovery_active=True,
        flip_raw_min=0.99,
    )
    assert best[0] == HEDGE_PEER_SYMBOL
    assert best[1] == TradeDirection.PUT
