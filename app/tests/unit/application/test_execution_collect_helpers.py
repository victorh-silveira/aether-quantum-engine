from types import SimpleNamespace

from src.application.services.orchestrator.execution_collect_helpers import mandatory_fallback_if_empty
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR


def test_helper_mandatory_fallback_if_empty_returns_existing_candidates():
    exec_mgr = SimpleNamespace(
        _trade_symbols=lambda: [ANCHOR],
        orch=SimpleNamespace(risk_manager=SimpleNamespace(consecutive_losses_linear=0)),
    )
    existing = [(ANCHOR, TradeDirection.CALL, {"trade_score": 0.6})]
    kept = mandatory_fallback_if_empty(
        exec_mgr,
        {},
        existing,
        mandatory=True,
        recovery_active=False,
        last_loss=None,
        skip_symbols=frozenset(),
        min_signal=0.5,
        min_val=0.5,
    )
    assert kept is existing
