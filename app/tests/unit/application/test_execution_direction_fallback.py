from src.application.services.execution_direction_fallback import build_mandatory_fallback_candidate
from src.domain.models.trade import TradeDirection


def test_build_mandatory_fallback_candidate_invalid_pool_still_picks():
    best = build_mandatory_fallback_candidate(
        ["OTC_SPC"],
        {
            "OTC_SPC": {
                "direction": TradeDirection.CALL,
                "metrics": {"trade_score": 0.55, "raw_prob": 0.72, "deploy_ok": True, "val_accuracy": 0.55},
            }
        },
        recovery_active=True,
        last_loss_symbol=None,
    )
    assert best is not None
    assert best[0] in {"OTC_SPC", "R_50"}


def test_build_mandatory_fallback_candidate_non_recovery_uses_raw():
    decisions = {
        "OTC_SPC": {
            "direction": None,
            "metrics": {
                "gate_reason": "direction_margin",
                "trade_score": 0.62,
                "raw_prob": 0.62,
                "deploy_ok": True,
                "val_accuracy": 0.55,
            },
        },
    }
    best = build_mandatory_fallback_candidate(["OTC_SPC"], decisions, recovery_active=False, last_loss_symbol=None)
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_candidate_recovery_prefers_alt_symbol():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.40, "raw_prob": 0.40, "deploy_ok": True, "val_accuracy": 0.55},
        },
        "R_75": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.70, "raw_prob": 0.35, "deploy_ok": True, "val_accuracy": 0.60},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["OTC_SPC", "R_75"], decisions, recovery_active=True, last_loss_symbol="OTC_SPC"
    )
    assert best is not None
    assert best[0] == "R_75"


def test_build_mandatory_fallback_candidate_recovery_without_loss_symbol():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.66, "raw_prob": 0.66, "deploy_ok": True, "val_accuracy": 0.55},
        },
    }
    best = build_mandatory_fallback_candidate(["OTC_SPC"], decisions, recovery_active=True, last_loss_symbol=None)
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_skips_blocked_gate():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.80, "raw_prob": 0.80, "deploy_ok": False, "val_accuracy": 0.55},
        },
        "R_75": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.61, "raw_prob": 0.39, "deploy_ok": True, "val_accuracy": 0.55},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["OTC_SPC", "R_75"], decisions, recovery_active=False, last_loss_symbol=None
    )
    assert best is not None
    assert best[0] == "R_75"


def test_build_mandatory_fallback_respects_min_signal():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.40, "raw_prob": 0.52, "deploy_ok": True, "val_accuracy": 0.55},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["OTC_SPC"], decisions, recovery_active=False, last_loss_symbol=None, min_signal=0.70
    )
    assert best is None


def test_build_mandatory_fallback_respects_min_val():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.70, "raw_prob": 0.70, "deploy_ok": True, "val_accuracy": 0.40},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["OTC_SPC"], decisions, recovery_active=False, last_loss_symbol=None, min_val=0.55
    )
    assert best is None


def test_build_mandatory_fallback_skip_symbols():
    decisions = {
        "OTC_SPC": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.80, "raw_prob": 0.80, "deploy_ok": True, "val_accuracy": 0.60},
        },
        "R_75": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.70, "raw_prob": 0.30, "deploy_ok": True, "val_accuracy": 0.60},
        },
    }
    best = build_mandatory_fallback_candidate(
        ["OTC_SPC", "R_75"],
        decisions,
        recovery_active=True,
        last_loss_symbol="OTC_SPC",
        skip_symbols=frozenset({"OTC_SPC"}),
    )
    assert best is not None
    assert best[0] == "R_75"


def test_build_mandatory_fallback_empty_decisions():
    assert build_mandatory_fallback_candidate(["OTC_SPC"], {}, recovery_active=True, last_loss_symbol="OTC_SPC") is None


def test_build_mandatory_fallback_recovery_uses_dl():
    decisions = {
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "trade_score": 0.58,
                "raw_prob": 0.58,
                "calibrated_prob": 0.58,
                "deploy_ok": True,
                "val_accuracy": 0.55,
            },
        },
    }
    best = build_mandatory_fallback_candidate(["R_75"], decisions, recovery_active=True, last_loss_symbol="OTC_SPC")
    assert best is not None
    assert best[1] == TradeDirection.CALL


def test_build_mandatory_fallback_keeps_dl_side_after_put_loss():
    decisions = {
        "R_75": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "trade_score": 0.62,
                "raw_prob": 0.38,
                "calibrated_prob": 0.38,
                "deploy_ok": True,
                "val_accuracy": 0.55,
            },
        },
    }
    best = build_mandatory_fallback_candidate(["R_75"], decisions, recovery_active=True, last_loss_symbol="OTC_SPC")
    assert best is not None
    assert best[1] == TradeDirection.PUT


def test_build_mandatory_fallback_keeps_dl_side_after_call_loss():
    decisions = {
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "trade_score": 0.62,
                "raw_prob": 0.62,
                "calibrated_prob": 0.62,
                "deploy_ok": True,
                "val_accuracy": 0.55,
            },
        },
    }
    best = build_mandatory_fallback_candidate(["R_75"], decisions, recovery_active=True, last_loss_symbol="OTC_SPC")
    assert best is not None
    assert best[1] == TradeDirection.CALL
