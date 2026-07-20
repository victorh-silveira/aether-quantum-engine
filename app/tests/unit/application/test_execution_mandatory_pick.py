from unittest.mock import patch

from src.application.services.execution_direction_fallback import build_mandatory_fallback_candidate
from src.application.services.execution_mandatory_pick import (
    pick_absolute_mandatory_candidate,
    pick_best_mandatory_candidate,
)
from src.application.services.meta_payoff_veto_gate import apply_meta_payoff_negative_zscore_veto
from src.domain.models.trade import TradeDirection
from tests.unit.application.universal_regime_metrics import bear_put_metrics


def test_pick_absolute_mandatory_skips_training_symbols():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "gate_reason": "training", "trade_score": 0.70, "raw_prob": 0.40},
        },
    }
    picked = pick_absolute_mandatory_candidate(
        ["R_10"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert picked is None


def test_pick_best_mandatory_survives_vetoed_entry_with_nulled_scores():
    bull_metrics = {
        "execute": True,
        "deploy_ok": True,
        "raw_prob": 0.70,
        "trade_score": 0.70,
        "val_accuracy": 0.72,
        "edge": 1.40,
        "trend_direction": "CALL",
        "indicators": {"hurst": 0.55, "adx": 0.30, "vol_ratio": 1.0},
    }
    bear_metrics = {
        "execute": True,
        "deploy_ok": True,
        "raw_prob": 0.51,
        "trade_score": 0.51,
        "val_accuracy": 0.71,
        "trend_direction": "PUT",
    }
    apply_meta_payoff_negative_zscore_veto(bear_metrics)
    decisions = {
        "R_10": {"direction": TradeDirection.CALL, "metrics": bull_metrics},
        "R_50": {"direction": TradeDirection.PUT, "metrics": bear_metrics},
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_50"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}


def test_pick_best_mandatory_prefers_strong_r50_over_weak_r10():
    decisions = {
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.80, "val_accuracy": 0.71, "raw_prob": 0.80},
        },
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": False, "trade_score": 0.51, "val_accuracy": 0.0, "raw_prob": 0.51},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_50"],
        decisions,
        recovery_active=False,
        last_loss_symbol=None,
        last_loss_direction=None,
    )
    assert picked is not None
    assert picked[0] == "R_50"


def test_pick_best_mandatory_recovery_prefers_market_rank_over_loss_direction():
    decisions = {
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "trade_score": 0.80,
                "val_accuracy": 0.55,
                "raw_prob": 0.20,
                "deploy_ok": True,
                "execute": True,
            },
        },
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "trade_score": 0.44,
                "val_accuracy": 0.55,
                "raw_prob": 0.80,
                "deploy_ok": False,
                "gate_reason": "deploy",
                "execute": True,
            },
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_50"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is not None
    assert picked[0] == "R_50"
    assert picked[1] == TradeDirection.PUT


def test_pick_best_mandatory_aligned_skips_low_val_accuracy():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.62, "val_accuracy": 0.40, "raw_prob": 0.58},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is None


def test_pick_absolute_mandatory_skips_below_signal_floor():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.20, "raw_prob": 0.44},
        },
    }
    picked = pick_absolute_mandatory_candidate(
        ["R_10"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
        min_signal=0.50,
        min_val=0.50,
    )
    assert picked is None


def test_pick_absolute_mandatory_skips_weak_recovery_signal():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.0, "val_accuracy": 0.55},
        },
    }
    picked = pick_absolute_mandatory_candidate(
        ["R_10"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
        min_signal=0.50,
        min_val=0.50,
    )
    assert picked is None


def test_pick_absolute_mandatory_skips_low_val_accuracy_in_recovery():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": False, "trade_score": 0.60, "raw_prob": 0.60, "val_accuracy": 0.40},
        },
    }
    picked = pick_absolute_mandatory_candidate(
        ["R_10"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
        min_signal=0.50,
        min_val=0.50,
    )
    assert picked is None


def test_pick_best_skips_aligned_candidate_below_min_signal():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.30, "val_accuracy": 0.55},
        },
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.65, "val_accuracy": 0.55, "raw_prob": 0.42},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_50"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}


def test_pick_best_skips_aligned_candidate_below_recovery_thresholds():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.30, "val_accuracy": 0.40, "raw_prob": 0.52},
        },
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.65, "val_accuracy": 0.55, "raw_prob": 0.42},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_50"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
        min_signal=0.45,
        min_val=0.50,
    )
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}


def test_build_mandatory_fallback_legacy_path_when_market_rank_empty():
    with patch(
        "src.application.services.execution_direction_fallback.pick_best_mandatory_candidate",
        return_value=None,
    ):
        best = build_mandatory_fallback_candidate(
            ["R_10"],
            {
                "R_10": {
                    "direction": TradeDirection.PUT,
                    "metrics": bear_put_metrics(trade_score=0.55, raw_prob=0.42, calibrated_prob=0.42),
                }
            },
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
        )
    assert best is not None
    assert best[1] == TradeDirection.PUT
