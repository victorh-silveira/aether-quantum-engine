from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.orchestrator.execution_collect_helpers import log_execution_decision
from src.domain.models.trade import TradeDirection


def test_log_execution_decision_direct():
    exec_mgr = SimpleNamespace(logger=MagicMock())
    best = (
        "R_10",
        TradeDirection.CALL,
        {
            "val_accuracy": 0.55,
            "raw_prob": 0.52,
            "indicators": {"hurst": 0.56, "adx": 31.0},
            "call_votes": 4,
            "put_votes": 2,
        },
    )
    log_execution_decision(exec_mgr, "C0001", best, [best], 0.55)
    assert exec_mgr.logger.info.call_count >= 2
    payloads = [c.args[1] for c in exec_mgr.logger.info.call_args_list]
    assert payloads[0].startswith("[GATES] || LOSS_CLF:")
    assert any(str(p).startswith("[IND] ||") for p in payloads)
