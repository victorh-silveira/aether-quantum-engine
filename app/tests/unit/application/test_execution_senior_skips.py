"""Testes das salvaguardas que bloqueiam, sem sobrescrever, a direcao."""

from src.application.services.execution_senior_skips import apply_senior_execution_skips
from src.domain.models.trade import TradeDirection


def test_apply_senior_execution_skips_passes_without_active_guard():
    direction, blocked = apply_senior_execution_skips(TradeDirection.CALL, {})
    assert direction == TradeDirection.CALL
    assert blocked is False


def test_apply_senior_execution_skips_blocks_independent_of_legacy_flag():
    metrics = {"indicators": {"adx": 0.10, "bb_width": 0.02}, "trend_direction": "PUT"}
    cfg = {"skip_chop_congestion": True}
    direction, blocked = apply_senior_execution_skips(TradeDirection.CALL, metrics, exec_cfg=cfg)
    assert direction == TradeDirection.CALL
    assert blocked is True
    assert metrics["skip_reason"] == "chop_congestion"


def test_should_skip_explosion_discord_blocks_opposing_candle():
    from src.application.services.execution_senior_skips import should_skip_explosion_discord

    assert should_skip_explosion_discord({}, TradeDirection.CALL) is False

    metrics = {
        "scale_micro_regime": "explosion",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.02,
    }
    assert should_skip_explosion_discord(metrics, TradeDirection.CALL) is True
    assert metrics["skip_reason"] == "explosion_discord"

    metrics_edge = {
        "scale_micro_regime": "explosion",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.09,
    }
    assert should_skip_explosion_discord(metrics_edge, TradeDirection.CALL) is False

    metrics_aligned = {
        "scale_micro_regime": "explosion",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "CALL",
        "cal_side_edge": 0.02,
    }
    assert should_skip_explosion_discord(metrics_aligned, TradeDirection.CALL) is False

    metrics_flip = {
        "scale_micro_regime": "explosion",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.02,
        "anti_trend_lock_flip": True,
    }
    assert should_skip_explosion_discord(metrics_flip, TradeDirection.CALL) is False


def test_apply_senior_execution_skips_blocks_explosion_discord_under_four_vetoes():
    metrics = {
        "scale_micro_regime": "explosion",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": "invalid",
    }
    cfg = {"four_market_vetoes": True}
    direction, blocked = apply_senior_execution_skips(TradeDirection.CALL, metrics, exec_cfg=cfg)
    assert direction == TradeDirection.CALL
    assert blocked is True
    assert metrics["skip_reason"] == "explosion_discord"


def test_should_skip_explosion_discord_configurable_and_pend_waiver():
    from src.application.services.execution_senior_skips import should_skip_explosion_discord

    metrics = {
        "scale_micro_regime": "explosion",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.02,
    }
    assert (
        should_skip_explosion_discord(metrics, TradeDirection.CALL, exec_cfg={"skip_explosion_discord": False}) is False
    )

    cfg_floor = {"explosion_discord_min_edge": 0.01}
    assert should_skip_explosion_discord(metrics, TradeDirection.CALL, exec_cfg=cfg_floor) is False

    metrics_pend = {
        "scale_micro_regime": "explosion",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.02,
        "pending_loss_total": 5.0,
    }
    assert should_skip_explosion_discord(metrics_pend, TradeDirection.CALL, exec_cfg={"cover_enabled": True}) is False
