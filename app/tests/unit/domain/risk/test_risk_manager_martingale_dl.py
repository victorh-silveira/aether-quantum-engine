import math

import pytest

from src.domain.risk.risk_manager import RiskManager


def test_martingale_uses_recorded_loss_stake_as_seed(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.record_contract_stake(1, 10.83)
    rm.register_result(-10.83, 1, "R_100")
    stake = rm.calculate_stake(
        10820.0,
        "R_10",
        conviction=0.62,
        dl_metrics={"execute": True, "trade_score": 0.62, "val_accuracy": 0.55},
    )
    cover = (10.83 + 10.83 * 0.95) / 0.95
    assert stake == pytest.approx(math.ceil(cover * 100) / 100, abs=0.5)


def test_martingale_requires_conviction_floor_with_pending(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 50.0
    assert not rm._martingale_allowed(
        "R_50",
        0.40,
        dl_metrics={
            "gate_reason": "deploy",
            "val_brier": 0.35,
            "deploy_ok": False,
            "val_accuracy": 0.40,
            "trade_score": 0.40,
            "execute": False,
        },
        order_direction="PUT",
    )
    assert rm._martingale_allowed(
        "R_75",
        0.56,
        dl_metrics={
            "gate_reason": "brier",
            "val_brier": 0.40,
            "val_accuracy": 0.55,
            "trade_score": 0.60,
            "execute": True,
        },
        order_direction="CALL",
        last_loss_symbol="R_50",
        last_loss_direction="CALL",
    )


def test_martingale_skips_weak_recovery_signal(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 10.0
    rm.last_loss_stake = 4.62
    stake = rm.calculate_stake(
        1150.0,
        "R_75",
        conviction=0.0,
        cycle_id=5,
        dl_metrics={"execute": False, "trade_score": 0.0, "val_accuracy": 0.51, "recovery_forced": True},
    )
    assert stake == 0.0


def test_martingale_rejects_marginal_raw_side_below_sizing_conviction(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 10.0
    assert not rm._martingale_allowed(
        "R_75",
        0.51,
        dl_metrics={
            "execute": False,
            "trade_score": 0.0,
            "raw_prob": 0.49,
            "val_accuracy": 0.59,
        },
    )


def test_martingale_allowed_on_strong_raw_side_with_zero_score(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 10.0
    rm.last_loss_stake = 4.62
    assert rm._martingale_allowed(
        "R_75",
        0.0,
        dl_metrics={
            "execute": False,
            "trade_score": 0.0,
            "raw_prob": 0.62,
            "val_accuracy": 0.55,
        },
    )


def test_martingale_allowed_without_pending(kelly_config):
    rm = RiskManager(kelly_config)
    assert not rm._martingale_allowed("R_25", 0.50, dl_metrics={"execute": True, "trade_score": 0.60})


def test_martingale_stake_grows_with_pending_loss(kelly_config):
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-10.0, 1, "R_50")
    dl_metrics = {"execute": True, "trade_score": 0.60, "val_accuracy": 0.55}
    first = rm.calculate_stake(1000.0, "R_75", conviction=0.56, dl_metrics=dl_metrics)
    rm.active_contract_ids = [2]
    rm.register_result(-20.0, 2, "R_75")
    second = rm.calculate_stake(970.0, "R_75", conviction=0.56, dl_metrics=dl_metrics)
    assert second > first


def test_symbol_loss_cooldown_records_direction(kelly_config):
    kelly_config["kelly"]["symbol_loss_cooldown_cycles"] = 1
    rm = RiskManager(kelly_config)
    rm.active_contract_ids = [1]
    rm.register_result(-5.0, 1, "R_50", direction="PUT")
    assert rm.last_loss_direction == "PUT"


def test_recovery_symbol_loss_streak_increments_on_martingale_loss(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 5.0
    rm.active_contract_ids = [101]
    rm.register_result(-5.0, 101, symbol="R_50", current_tick=1, direction="CALL")
    assert rm.recovery_symbol_loss_streak.get("R_50") == 1
    rm.pending_loss["R_50"] = 5.0
    rm.active_contract_ids = [102]
    rm.register_result(-8.0, 102, symbol="R_50", current_tick=2, direction="CALL")
    assert rm.recovery_symbol_loss_streak.get("R_50") == 2


def test_recovery_symbol_loss_streak_resets_after_full_recovery(kelly_config):
    rm = RiskManager(kelly_config)
    rm.pending_loss["R_50"] = 5.0
    rm.active_contract_ids = [101]
    rm.register_result(-5.0, 101, symbol="R_50", current_tick=1, direction="CALL")
    rm.active_contract_ids = [102]
    rm.register_result(12.0, 102, symbol="R_75", current_tick=2, direction="CALL")
    assert rm.recovery_symbol_loss_streak == {}
