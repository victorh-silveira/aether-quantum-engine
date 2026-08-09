"""Cobertura residual da visao multi-escala (parte 2)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.application.services.deep_learning.dl_cycle_log import log_dl_cycle_summary
from src.domain.models.trade import TradeDirection


def test_log_dl_cycle_with_scale_audit_dedupes():
    logger = MagicMock()
    orch = SimpleNamespace(_active_cycle_id=1, config={"data_handler": {"micro_granularity": 120}})
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "conviction": 0.70,
                "execute": True,
                "val_accuracy": 0.55,
                "cal_margin": 0.05,
                "scale_audit": "SCALE || MACRO=CALL MICRO=CALL MINI=CALL MILI=CALL agree=4/4 discord=False",
            },
        },
    }
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    scale_calls = [c for c in logger.debug.call_args_list if "SCALE" in str(c)]
    assert logger.info.call_count == 1
    assert len(scale_calls) == 2
