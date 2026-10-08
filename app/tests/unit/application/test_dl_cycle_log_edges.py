from unittest.mock import patch

from src.application.services.deep_learning.dl_cycle_brief import (
    _abstain_detail,
    _brief_cycle_counts,
    _format_brief_token,
    build_dl_cycle_brief,
)
from src.domain.models.trade import TradeDirection


def test_build_dl_cycle_brief_exec_and_blocked():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "conviction": 0.75,
                "execute": True,
                "raw_prob": 0.25,
                "calibrated_prob": 0.25,
                "direction_margin": 0.25,
                "deploy_ok": True,
            },
        },
        "R_50": {"direction": None, "metrics": {"gate_reason": "predict_error", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "exec R_10:PUT m=0.25 cal=0.25" in line
    assert "1 bloq" in line


def test_build_dl_cycle_brief_all_training():
    decisions = {
        "R_10": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert line == "DL | TREINO INICIAL | 2 modelo(s) em treinamento | trades suspensos"


def test_build_dl_cycle_brief_exec_with_training():
    decisions = {
        "R_10": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "conviction": 0.75,
                "execute": True,
                "raw_prob": 0.25,
                "calibrated_prob": 0.25,
                "direction_margin": 0.25,
                "deploy_ok": True,
            },
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "exec R_50:PUT m=0.25 cal=0.25" in line
    assert "1 treinando" in line


def test_build_dl_cycle_brief_all_blocked_without_raw_prob():
    decisions = {
        "R_10": {"direction": None, "metrics": {"gate_reason": "predict_error", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "aguardando sinal" in line


def test_build_dl_cycle_brief_partial_no_data():
    decisions = {
        "R_10": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert line == "DL | sem exec | 1 sem dados | 1 treinando"


def test_build_dl_cycle_brief_bias_tokens():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "raw_prob": 0.42,
                "trade_score": 0.58,
                "deploy_ok": True,
                "dl_direction": "PUT",
                "exec_direction": "CALL",
                "direction_hint": "trend_bias",
            },
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "bias" in line


def test_abstain_detail_mixed_blocked_and_valid():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"raw_prob": 0.62, "execute": True, "deploy_ok": True},
        },
        "R_50": {"direction": None, "metrics": {"gate_reason": "predict_error", "raw_prob": 0.52, "execute": False}},
    }
    detail = _abstain_detail(decisions)
    assert "R_50" in detail


def test_format_brief_token_with_suffix():
    token = _format_brief_token("R_10", TradeDirection.CALL, margin=0.05, cal=0.55, suffix=":edge")
    assert token.endswith(":edge")
    assert "m=0.05" in token
    assert "cal=0.55" in token


def test_build_dl_cycle_brief_former_neutral_resolves_direction():
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {
                "execute": True,
                "gate_reason": None,
                "calibration_mode": "calibrated",
                "calibrated_prob": 0.50,
                "raw_prob": 0.52,
                "deploy_ok": True,
            },
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "exec R_10" in line
    assert "CALL" in line or "PUT" in line


def test_build_dl_cycle_brief_partial_blocked_tail():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": True,
                "raw_prob": 0.62,
                "calibrated_prob": 0.62,
                "direction_margin": 0.12,
                "deploy_ok": True,
            },
        },
        "R_50": {"direction": None, "metrics": {"gate_reason": "predict_error", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "1 bloq" in line
    assert "m=0.12" in line


def test_brief_cycle_counts_marks_missing_direction_as_blocked():
    exec_tokens, bias_tokens, blocked, no_data, training = _brief_cycle_counts(
        {
            "R_10": {"direction": None, "metrics": {"execute": True, "deploy_ok": True}},
        }
    )
    assert blocked == 1
    assert not exec_tokens
    assert not bias_tokens


def test_build_dl_cycle_brief_blocked_only_tail():
    decisions = {
        "R_10": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_50": {"direction": None, "metrics": {"execute": False, "deploy_ok": True}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "1 bloq" in line


def test_build_dl_cycle_brief_counts_block_when_direction_infer_fails():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"gate_reason": "predict_error", "execute": False, "trade_score": 0.52},
        },
    }
    with patch("src.application.services.deep_learning.dl_cycle_brief.infer_dl_direction", return_value=None):
        line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "aguardando sinal" in line


def test_build_dl_cycle_brief_returns_blocked_count_when_partial_training():
    decisions = {
        "R_10": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert line == "DL | sem exec | 1 sem dados | 1 treinando"
