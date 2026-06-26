from unittest.mock import patch

from src.application.services.deep_learning.dl_cycle_brief import build_dl_cycle_brief, build_dl_cycle_brief_key
from src.domain.models.trade import TradeDirection


def test_build_dl_cycle_brief_key():
    decisions = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"execute": True, "conviction": 0.65}},
        "R_75": {"direction": None, "metrics": {"execute": False, "gate_reason": "data"}},
        "R_100": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    key = build_dl_cycle_brief_key(decisions, recovery_active=False)
    assert "R_50:CALL" in key
    assert "c=" not in key
    assert "1 bloq" in key

    decisions_infer = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"execute": False, "trade_score": 0.52}},
    }
    with patch("src.application.services.deep_learning.dl_cycle_brief.infer_dl_direction", return_value=None):
        key_infer = build_dl_cycle_brief_key(decisions_infer, recovery_active=False)
    assert "sem exec" in key_infer

    decisions_train = {"R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}}}
    key_train = build_dl_cycle_brief_key(decisions_train, recovery_active=False)
    assert "TREINO INICIAL" in key_train

    decisions_nd = {"R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}}}
    key_nd = build_dl_cycle_brief_key(decisions_nd, recovery_active=False)
    assert "sem dados" in key_nd

    decisions_bp = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"gate_reason": "edge", "execute": False, "raw_prob": 0.58},
        }
    }
    key_bp = build_dl_cycle_brief_key(decisions_bp, recovery_active=False)
    assert "sinal R_50:CALL:edge" in key_bp


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
        "R_75": {"direction": None, "metrics": {"gate_reason": "predict_error", "execute": False}},
    }
    key_other = build_dl_cycle_brief_key(decisions_other, recovery_active=False)
    assert "1 bloq" in key_other
    assert "1 treinando" in key_other


def test_build_dl_cycle_brief_key_blocked_count_fallback():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "predict_error", "execute": False}},
    }
    key = build_dl_cycle_brief_key(decisions, recovery_active=False)
    assert "aguardando sinal" in key or "bloq" in key


def test_build_dl_cycle_brief_key_bias_branch():
    decisions = {
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": False,
                "raw_prob": 0.42,
                "dl_direction": "PUT",
                "exec_direction": "CALL",
                "direction_hint": "trend_bias",
                "trade_score": 0.58,
                "deploy_ok": True,
            },
        },
    }
    key = build_dl_cycle_brief_key(decisions, recovery_active=False)
    assert "bias" in key


def test_build_dl_cycle_brief_key_strips_raw_from_all_blocked_detail():
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {"gate_reason": "predict_error", "raw_prob": 0.52, "execute": False},
        },
    }
    key = build_dl_cycle_brief_key(decisions, recovery_active=False)
    assert "r0.52" not in key
    assert "aguardando sinal" in key


def test_build_dl_cycle_brief_key_strips_raw_from_blocked_detail():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "edge", "execute": False, "raw_prob": 0.58}},
    }
    key = build_dl_cycle_brief_key(decisions, recovery_active=False)
    assert "r0.58" not in key

    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "predict_error", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "aguardando sinal" in line or "bloq" in line
