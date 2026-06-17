from unittest.mock import patch

from src.application.services.deep_learning.dl_cycle_log import (
    build_dl_cycle_brief_key,
)
from src.domain.models.trade import TradeDirection


def test_build_dl_cycle_brief_key():
    decisions = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"execute": True, "conviction": 0.65}},
        "R_75": {"direction": None, "metrics": {"execute": False, "gate_reason": "conviction"}},
        "R_100": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    key = build_dl_cycle_brief_key(decisions, recovery_active=False)
    assert "R_50:CALL" in key
    assert "c=" not in key
    assert "1 bloq" in key

    decisions_infer = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"execute": False, "trade_score": 0.52}},
    }
    with patch("src.application.services.deep_learning.dl_cycle_log.infer_dl_direction", return_value=None):
        key_infer = build_dl_cycle_brief_key(decisions_infer, recovery_active=False)
    assert "sem exec" in key_infer

    decisions_train = {"R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}}}
    key_train = build_dl_cycle_brief_key(decisions_train, recovery_active=False)
    assert "TREINO INICIAL" in key_train

    decisions_nd = {"R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}}}
    key_nd = build_dl_cycle_brief_key(decisions_nd, recovery_active=False)
    assert "sem dados" in key_nd

    decisions_bp = {"R_50": {"direction": None, "metrics": {"gate_reason": "edge", "execute": False, "raw_prob": 0.58}}}
    key_bp = build_dl_cycle_brief_key(decisions_bp, recovery_active=False)
    assert "R_50:edge" in key_bp
    assert "r0.58" not in key_bp


def test_build_dl_cycle_brief_key_no_data_and_blocked_mixed():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
    }
    key = build_dl_cycle_brief_key(decisions, recovery_active=False)
    assert "1 sem dados" in key
    assert "1 treinando" in key

    decisions_other = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "confidence", "execute": False, "raw_prob": 0.58}},
    }
    key_other = build_dl_cycle_brief_key(decisions_other, recovery_active=False)
    assert "1 bloq" in key_other
    assert "1 treinando" in key_other
