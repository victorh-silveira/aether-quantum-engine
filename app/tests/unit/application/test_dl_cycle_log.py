import logging

from src.application.services.deep_learning.dl_cycle_brief import _abstain_detail, _format_bias_token
from src.application.services.deep_learning.dl_cycle_log import (
    _best_directional_signal,
    build_dl_cycle_summary,
    log_dl_cycle_summary,
)
from src.domain.models.trade import TradeDirection


def test_build_dl_cycle_summary_sem_dir_without_raw_prob():
    decisions = {
        "OTC_SPC": {
            "direction": None,
            "metrics": {"gate_reason": "data", "execute": False},
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "OTC_SPC:data" in line


def test_build_dl_cycle_summary_normal_exec():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "conviction": 0.71,
                "execute": True,
                "raw_prob": 0.71,
                "val_accuracy": 0.50,
                "deploy_ok": True,
            },
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "DL | NORMAL" in line
    assert "OTC_SPC:PUT:0.71" in line


def test_build_dl_cycle_summary_shows_bias_on_inversion():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "execute": True,
                "raw_prob": 0.42,
                "trade_score": 0.58,
                "val_accuracy": 0.60,
                "deploy_ok": True,
                "dl_direction": "PUT",
                "exec_direction": "CALL",
                "direction_hint": "exhaustion_flip",
            },
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "bias=[" in line
    assert "exhaustion_flip" in line


def test_build_dl_cycle_summary_recovery_exec_token():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "conviction": 0.55,
                "execute": True,
                "raw_prob": 0.55,
                "val_accuracy": 0.60,
                "deploy_ok": True,
            },
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=True, pending_loss_total=50.0)
    assert ":rec" in line


def test_build_dl_cycle_summary_training_tokens():
    decisions = {
        "OTC_SPC": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "treino=[OTC_SPC,R_50]" in line


def test_abstain_detail_skips_training_symbols():
    decisions = {
        "OTC_SPC": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
    }
    detail = _abstain_detail(decisions)
    assert detail == "R_50:data"


def test_best_directional_signal_skips_training():
    decisions = {
        "OTC_SPC": {"direction": TradeDirection.CALL, "metrics": {"gate_reason": "training", "trade_score": 0.9}},
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": True, "trade_score": 0.61, "raw_prob": 0.39},
        },
    }
    assert _best_directional_signal(decisions) == ("R_50", 0.61)


def test_best_directional_signal_returns_none_when_all_skipped():
    decisions = {
        "OTC_SPC": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
        "R_50": {"direction": None, "metrics": {"gate_reason": "predict_error", "execute": False}},
    }
    assert _best_directional_signal(decisions) is None


def test_log_dl_cycle_summary_logs_info_without_orch(caplog):
    logger = logging.getLogger("test_dl_cycle_log_no_orch")
    logger.setLevel(logging.DEBUG)
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {"conviction": 0.70, "execute": True, "raw_prob": 0.70, "deploy_ok": True},
        },
    }
    with caplog.at_level(logging.DEBUG):
        log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0)
    assert any(
        r.levelname == "INFO" and "[CLUSTER]" in r.message and "OTC_SPC: CALL" in r.message for r in caplog.records
    )


def test_log_dl_cycle_summary(caplog):
    logger = logging.getLogger("test_dl_cycle_log")
    logger.setLevel(logging.DEBUG)
    decisions = {
        "OTC_SPC": {
            "direction": None,
            "metrics": {"gate_reason": "predict_error", "execute": False},
        }
    }
    with caplog.at_level(logging.DEBUG):
        log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0)
    assert "DL | NORMAL" in caplog.text
    assert any(
        r.levelname == "INFO" and "[CLUSTER]" in r.message and "PREDICT_ERROR" in r.message for r in caplog.records
    )


def test_format_bias_token_flip_and_hint_only():
    flip = _format_bias_token(
        "OTC_SPC",
        {
            "direction": TradeDirection.PUT,
            "metrics": {
                "dl_direction": "PUT",
                "exec_direction": "CALL",
                "direction_hint": "trend_bias",
                "trade_score": 0.62,
            },
        },
    )
    assert flip is not None
    assert "->CALL" in flip
    hint_only = _format_bias_token(
        "OTC_SPC",
        {
            "direction": TradeDirection.CALL,
            "metrics": {
                "dl_direction": "CALL",
                "exec_direction": "CALL",
                "direction_hint": "exhaustion_flip",
                "trade_score": 0.55,
            },
        },
    )
    assert hint_only is not None
    assert "exhaustion_flip" in hint_only


def test_build_dl_cycle_summary_bypass_and_recovery_tokens():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "conviction": 0.71,
                "execute": True,
                "raw_prob": 0.71,
                "val_accuracy": 0.50,
                "bypass_val_acc": True,
                "deploy_ok": True,
            },
        },
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "conviction": 0.55,
                "execute": True,
                "raw_prob": 0.45,
                "val_accuracy": 0.60,
                "deploy_ok": True,
            },
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=True, pending_loss_total=100.0)
    assert ":bypass" in line
    assert ":rec" in line


def test_build_dl_cycle_summary_sem_dir_skip():
    decisions = {
        "R_51": {"direction": None, "metrics": {"execute": True, "deploy_ok": True}},
    }
    line = build_dl_cycle_summary(
        decisions,
        recovery_active=False,
        pending_loss_total=0.0,
    )
    assert "sem_dir" in line


def test_build_dl_cycle_summary_many_skip_tokens():
    decisions = {
        f"SYM{i}": {
            "direction": None,
            "metrics": {"gate_reason": "predict_error", "execute": False},
        }
        for i in range(7)
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "+2" in line
