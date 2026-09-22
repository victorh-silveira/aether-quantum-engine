"""Testes das salvaguardas que bloqueiam, sem sobrescrever, a direcao."""

from src.application.services.execution_senior_skips import apply_senior_execution_skips
from src.domain.models.trade import TradeDirection


def test_apply_senior_execution_skips_passes_without_active_guard():
    direction, blocked = apply_senior_execution_skips(TradeDirection.CALL, {})
    assert direction == TradeDirection.CALL
    assert blocked is False


def test_apply_senior_execution_skips_blocks_independent_of_legacy_flag():
    metrics = {"indicators": {"adx": 0.10, "bb_width": 0.02}, "trend_direction": "PUT"}
    cfg = {"skip_chop_congestion": True}
    direction, blocked = apply_senior_execution_skips(TradeDirection.CALL, metrics, exec_cfg=cfg)
    assert direction == TradeDirection.CALL
    assert blocked is True
    assert metrics["skip_reason"] == "chop_congestion"
