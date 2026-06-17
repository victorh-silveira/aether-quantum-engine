import logging

from src.application.services.deep_learning.dl_cycle_log import (
    _abstain_detail,
    _best_directional_signal,
    build_dl_cycle_brief,
    build_dl_cycle_summary,
    log_dl_cycle_summary,
)
from src.domain.models.trade import TradeDirection


def test_build_dl_cycle_summary_confidence_raw():
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {"gate_reason": "confidence", "raw_prob": 0.52, "execute": False},
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
        "R_50": {"direction": None, "metrics": {"gate_reason": "confidence", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "R_100:PUT c=0.75" in line
    assert "1 bloq" in line


def test_build_dl_cycle_brief_all_training():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert line == "DL | TREINO INICIAL | 2 modelo(s) em treinamento | trades suspensos"


def test_build_dl_cycle_brief_mixed_training_and_weak_exec():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {"conviction": 0.53, "execute": False, "gate_reason": "conviction"},
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "sinal R_75:CALL c=0.53" in line
    assert "1 treinando" in line


def test_build_dl_cycle_brief_exec_with_training():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_100": {
            "direction": TradeDirection.PUT,
            "metrics": {"conviction": 0.75, "execute": True, "val_accuracy": 1.0},
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "R_100:PUT c=0.75" in line
    assert "1 treinando" in line


def test_build_dl_cycle_summary_training_tokens():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_summary(decisions, recovery_active=False, pending_loss_total=0.0)
    assert "treino=[R_50,R_75]" in line


def test_abstain_detail_skips_training_symbols():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "confidence", "raw_prob": 0.52}},
    }
    detail = _abstain_detail(decisions)
    assert detail == "R_75:r0.52:confidence"


def test_build_dl_cycle_brief_weak_signal_shows_candidate():
    decisions = {
        "R_100": {
            "direction": TradeDirection.PUT,
            "metrics": {"gate_reason": "confidence", "execute": False, "trade_score": 0.50, "raw_prob": 0.50},
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "sinal R_100:PUT c=0.50" in line


def test_build_dl_cycle_brief_all_blocked_without_raw_prob():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "edge", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "R_50:edge" in line


def test_build_dl_cycle_brief_recovery_weak_signal_shows_candidate():
    decisions = {
        "R_100": {
            "direction": TradeDirection.PUT,
            "metrics": {"gate_reason": "confidence", "execute": False, "trade_score": 0.50, "raw_prob": 0.50},
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=True)
    assert "sinal R_100:PUT c=0.50" in line


def test_build_dl_cycle_brief_recovery_all_blocked_shows_abstain():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "conviction", "execute": False, "raw_prob": 0.52}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "edge", "execute": False, "raw_prob": 0.48}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=True)
    assert "aguardando sinal" in line


def test_best_directional_signal_skips_training():
    decisions = {
        "R_50": {"direction": TradeDirection.CALL, "metrics": {"gate_reason": "training", "trade_score": 0.9}},
        "R_75": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.61, "raw_prob": 0.39},
        },
    }
    assert _best_directional_signal(decisions) == ("R_75", 0.61)


def test_best_directional_signal_skips_without_dl_direction():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
        "R_75": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.61, "raw_prob": 0.39},
        },
    }
    assert _best_directional_signal(decisions) == ("R_75", 0.61)


def test_best_directional_signal_returns_none_when_all_skipped():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "conviction", "execute": False}},
    }
    assert _best_directional_signal(decisions) is None


def test_build_dl_cycle_brief_recovery_all_blocked_returns_detail():
    decisions = {
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "trade_score": 0.55,
                "raw_prob": 0.53,
                "val_accuracy": 0.67,
                "gate_reason": "confidence",
            },
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=True)
    assert "sinal R_75:CALL c=0.55" in line


def test_build_dl_cycle_brief_partial_no_data():
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "data", "execute": False}},
        "R_75": {"direction": None, "metrics": {"gate_reason": "training", "execute": False}},
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert line == "DL | sem exec | 1 sem dados | 1 treinando"


def test_log_dl_cycle_summary_logs_info_without_orch(caplog):
    logger = logging.getLogger("test_dl_cycle_log_no_orch")
    logger.setLevel(logging.DEBUG)
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"conviction": 0.70, "execute": True, "val_accuracy": 0.55},
        },
    }
    with caplog.at_level(logging.DEBUG):
        log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0)
    assert any(r.levelname == "INFO" and "exec R_50:CALL" in r.message for r in caplog.records)


def test_build_dl_cycle_brief_normal_all_blocked_shows_abstain():
    decisions = {
        "R_50": {
            "direction": None,
            "metrics": {"gate_reason": "edge", "execute": False, "raw_prob": 0.58},
        },
        "R_75": {
            "direction": None,
            "metrics": {"gate_reason": "brier", "execute": False, "raw_prob": 0.42},
        },
    }
    line = build_dl_cycle_brief(decisions, recovery_active=False)
    assert "aguardando sinal" in line
    assert "R_50:r0.58" in line


def test_log_dl_cycle_summary_skips_info_when_recovery_brief_empty(caplog):
    logger = logging.getLogger("test_dl_cycle_log_recovery_empty")
    logger.setLevel(logging.DEBUG)
    decisions = {
        "R_50": {"direction": None, "metrics": {"gate_reason": "conviction", "execute": False, "raw_prob": 0.52}},
    }
    with caplog.at_level(logging.DEBUG):
        log_dl_cycle_summary(logger, decisions, recovery_active=True, pending_loss_total=10.0)
    assert any(r.levelname == "INFO" and "aguardando sinal" in r.message for r in caplog.records)


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
    assert any(r.levelname == "INFO" and "aguardando sinal" in r.message for r in caplog.records)
