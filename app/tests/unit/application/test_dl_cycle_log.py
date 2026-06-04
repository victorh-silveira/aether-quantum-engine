import logging

from src.application.services.deep_learning.dl_cycle_log import (
    build_dl_cycle_brief,
    build_dl_cycle_summary,
    log_dl_cycle_summary,
)
from src.domain.models.trade import TradeDirection


def test_build_dl_cycle_summary_direction_margin_raw():
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {"gate_reason": "direction_margin", "raw_prob": 0.52, "execute": False},
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "sem_dir:r0.52" in line


def test_build_dl_cycle_summary_normal():
    decisions = {
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "conviction": 0.71,
                "execute": True,
                "val_accuracy": 0.50,
                "bypass_val_acc": True,
            },
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "conviction": 0.53,
                "execute": False,
                "gate_reason": "conviction",
                "val_accuracy": 0.43,
                "edge": 0.03,
                "raw_conviction": 0.54,
            },
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "DL | NORMAL" in line
    assert "R_50:PUT:0.71:v0.50:bypass" in line
    assert "R_75:conviction:r0.54:c0.53:e0.03:v0.43:b0.00:x0.00" in line


def test_build_dl_cycle_summary_recovery_and_truncation():
    decisions = {
        f"SYM{i}": {
            "direction": TradeDirection.CALL,
            "metrics": {"conviction": 0.5, "execute": False, "gate_reason": "edge", "val_accuracy": 0.5},
        }
        for i in range(7)
    }
    line = build_dl_cycle_summary(decisions, recovery_active=True, pending_loss_total=100.92)
    assert "RECOVERY pend=$101" in line
    assert "+2" in line


def test_build_dl_cycle_summary_recovery_exec_token():
    decisions = {
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "conviction": 0.55,
                "execute": True,
                "val_accuracy": 0.60,
                "bypass_val_acc": False,
            },
        },
    }
    line = build_dl_cycle_summary(decisions, recovery_active=True, pending_loss_total=50.0)
    assert ":rec" in line


def test_build_dl_cycle_brief_exec_and_blocked():
    decisions = {
        "R_100": {
            "direction": TradeDirection.PUT,
            "metrics": {"conviction": 0.75, "execute": True, "val_accuracy": 1.0},
        },
        "R_50": {"direction": None, "metrics": {"gate_reason": "direction_margin", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "R_100:PUT c=0.75" in line
    assert "1 bloq" in line


def test_log_dl_cycle_summary(caplog):
    logger = logging.getLogger("test_dl_cycle_log")
    logger.setLevel(logging.DEBUG)
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {"conviction": 0.0, "execute": False},
        }
    }
    with caplog.at_level(logging.DEBUG):
        log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0)
    assert "DL | NORMAL" in caplog.text
    assert "sem exec" in caplog.text
