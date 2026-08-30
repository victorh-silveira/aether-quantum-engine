"""Testes do catalogo minimo de atenuacao de sinal (escopo 1.1)."""

from unittest.mock import MagicMock

import pytest

from src.application.services.execution_signal_skip import (
    apply_signal_skip_gates,
    is_skip_signal_status,
    metrics_block_execution,
    parse_signal_skip_config,
)
from src.domain.models.trade import TradeDirection


def test_parse_signal_skip_from_ssot():
    cfg = parse_signal_skip_config({})
    assert cfg["enabled"] is True
    assert cfg["min_direction_margin"] == 0.005
    assert cfg["waive_margin_on_pending"] is True
    assert cfg["mini_pair_oppose_exec"] is False
    assert cfg["waive_mini_pair_min_margin"] == 0.0
    assert cfg["mini_pair_soft_kelly_mult"] == 0.75
    assert cfg["cal_margin_soft_kelly_mult"] == 0.75
    assert cfg["pending_dust"] == 0.25
    assert cfg["chop_pause_enabled"] is False
    assert cfg["chop_adx_max"] == 0.10
    assert cfg["chop_hurst_min"] == 0.45
    assert cfg["chop_hurst_max"] == 0.55
    assert cfg["chop_soft_kelly_mult"] == 0.75
    assert cfg["neg_edge_soft_kelly_mult"] == 0.55
    assert cfg["neg_edge_hard_skip"] is False
    assert cfg["neg_edge_soft_when_closed_candle_agree"] is True
    assert cfg["neg_edge_soft_min_edge"] == pytest.approx(-1.0)
    assert cfg["neg_edge_bootstrap_soft_kelly_mult"] == pytest.approx(0.25)
    assert cfg["neg_edge_deep_edge_floor"] == pytest.approx(-0.12)
    assert "direction_loss_lock_min" not in cfg
    assert "direction_loss_toxic_escape" not in cfg
    assert "calib_gray_margin_floor" not in cfg
    assert "calib_gray_soft_kelly_mult" not in cfg
    assert "calib_gray_max_stake_pct" not in cfg
    assert cfg["anti_loss_allow_candle_flip"] is True
    for k, v in (
        ("mini_pair_soft_kelly_mult", 0.0),
        ("cal_margin_soft_kelly_mult", 0.0),
        ("chop_soft_kelly_mult", 0.0),
        ("neg_edge_soft_kelly_mult", 0.0),
        ("neg_edge_soft_min_edge", 0.5),
        ("neg_edge_bootstrap_soft_kelly_mult", 0.0),
        ("neg_edge_deep_edge_floor", 0.5),
        ("anti_loss_p_loss_floor", -0.1),
        ("anti_loss_soft_kelly_mult", 0.0),
        ("anti_loss_min_candle_body", -0.1),
        ("anti_loss_live_confirm_min_body", 0.01),
        ("chop_hurst_max", 0.40),
    ):
        with pytest.raises(ValueError):
            parse_signal_skip_config({"chop_hurst_min": 0.60, k: v} if k == "chop_hurst_max" else {k: v})


def test_metrics_block_execution_covers_prefixed_skip_and_ready():
    assert is_skip_signal_status("SKIP:LOSS_CLF_VETO") is True
    assert is_skip_signal_status("SKIP:CAL_MARGIN") is True
    assert is_skip_signal_status("OK") is False
    assert metrics_block_execution({"signal_status": "SKIP:LOSS_CLF_VETO"}) is True
    assert metrics_block_execution({"execution_candidate_ready": False}) is True
    assert metrics_block_execution({"gate_reason": "cal_margin"}) is False
    assert metrics_block_execution({"signal_skip_reason": "mini_pair_oppose"}) is False
    assert metrics_block_execution({"gate_reason": "loss_clf_veto"}) is False
    assert metrics_block_execution({"execution_candidate_ready": True, "signal_status": "OPEN"}) is False


def _mini_pair_cfg(**overrides):
    cfg = parse_signal_skip_config(
        {
            "mini_pair_oppose_exec": True,
            "mini_pair_soft_kelly_mult": 0.55,
            "cal_margin_soft_kelly_mult": 0.55,
            "min_direction_margin": 0.022,
            "waive_margin_on_pending": False,
        }
    )
    cfg.update(overrides)
    return cfg


def test_mini_pair_oppose_always_soft_kelly():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.029,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, cfg=_mini_pair_cfg()) is False
    assert metrics.get("gate_reason") is None
    assert metrics.get("execution_candidate_ready") is not False
    assert metrics["signal_skip_waived"] == "mini_pair_soft"
    assert metrics["mini_pair_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.55)
    assert metrics_block_execution(metrics) is False


def test_mini_pair_oppose_strong_margin_soft_waive():
    metrics = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": 0.08,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, cfg=_mini_pair_cfg()) is False
    assert metrics.get("gate_reason") is None
    assert metrics.get("execution_candidate_ready") is not False
    assert metrics["signal_skip_waived"] == "mini_pair_soft"
    assert metrics["mini_pair_soft"] is True
    assert metrics["kelly_fraction_scale"] == 0.55


def test_cal_margin_soft_when_margin_weak_explore():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "direction_margin": 0.002,
        "pending_loss_total": 0.0,
        "kelly_fraction_scale": 1.0,
    }
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL) is False
    assert metrics.get("gate_reason") is None
    assert metrics["signal_skip_waived"] == "cal_margin_soft"
    assert metrics["cal_margin_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.75)
    assert metrics_block_execution(metrics) is False


def test_cal_margin_waived_when_pending_material():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "direction_margin": 0.002,
        "pending_loss_total": 27.0,
    }
    cfg = parse_signal_skip_config({"waive_margin_on_pending": True})
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, cfg=cfg) is False
    assert metrics.get("signal_skip_waived") == "cal_margin_pending"
    assert metrics.get("gate_reason") is None


def test_pending_map_fallback_and_bad_margin():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "PUT",
        "direction_margin": object(),
        "pending_loss_total": object(),
    }
    orch = MagicMock()
    del orch.risk_manager.pending_loss_total
    type(orch.risk_manager).pending_loss_total = property(lambda self: None)
    orch.risk_manager.pending_loss = {"R_10": 12.0}
    cfg = parse_signal_skip_config({"waive_margin_on_pending": True})
    assert apply_signal_skip_gates(metrics, TradeDirection.CALL, orch=orch, cfg=cfg) is False
    assert metrics.get("signal_skip_waived") == "cal_margin_pending"


def test_pending_total_edge_paths():
    from src.application.services.execution_signal_skip import _pending_total

    assert _pending_total({"pending_loss_total": "x"}, None) == 0.0
    orch_none_rm = MagicMock()
    orch_none_rm.risk_manager = None
    assert _pending_total({}, orch_none_rm) == 0.0

    orch_bad_fn = MagicMock()

    def _boom():
        raise ValueError("bad")

    orch_bad_fn.risk_manager.pending_loss_total = _boom
    assert _pending_total({}, orch_bad_fn) == 0.0

    orch_bad_map = MagicMock()
    orch_bad_map.risk_manager.pending_loss_total = "not_callable"
    orch_bad_map.risk_manager.pending_loss = {"R_10": object()}
    assert _pending_total({}, orch_bad_map) == 0.0

    orch_empty = MagicMock()
    orch_empty.risk_manager.pending_loss_total = "x"
    orch_empty.risk_manager.pending_loss = None
    assert _pending_total({}, orch_empty) == 0.0
