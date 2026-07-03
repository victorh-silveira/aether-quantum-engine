from src.application.services.execution_direction_compression_trap import (
    COMPRESSION_TRAP_BB_WIDTH_MAX,
    enforce_compression_trap_micro_bb_cohesion,
    micro_bb_width,
)
from src.application.services.execution_universal_regime_types import RegimeEvaluation, RegimeState
from src.domain.models.trade import TradeDirection


def test_micro_bb_width_defaults_when_missing():
    assert micro_bb_width({}) == 1.0


def test_enforce_compression_trap_micro_bb_cohesion_passes_when_bb_compressed():
    metrics = {"indicators": {"bb_width": COMPRESSION_TRAP_BB_WIDTH_MAX - 0.001}}
    evaluation = RegimeEvaluation(
        regime=RegimeState.COMPRESSION_TRAP,
        direction_inverted=True,
    )
    result = enforce_compression_trap_micro_bb_cohesion(
        TradeDirection.PUT,
        TradeDirection.CALL,
        metrics,
        evaluation,
    )
    assert result == TradeDirection.PUT
