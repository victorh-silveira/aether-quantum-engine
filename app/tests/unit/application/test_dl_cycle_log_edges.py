from unittest.mock import patch

from src.application.services.deep_learning.dl_cycle_log import build_dl_cycle_brief
from src.domain.models.trade import TradeDirection


def test_build_dl_cycle_brief_counts_block_when_direction_infer_fails():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"gate_reason": "confidence", "execute": False, "trade_score": 0.52},
        },
    }
    with patch("src.application.services.deep_learning.dl_cycle_log.infer_dl_direction", return_value=None):
        line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert line == "DL | sem exec | aguardando sinal | [R_50:confidence]"


def test_build_dl_cycle_brief_returns_blocked_count_when_partial_training():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "confidence", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert line == "DL | sem exec | 1 bloq | 1 treinando"
