"""Gate anti-loss seed + vela discordante do TCN."""

from __future__ import annotations

import pytest

from src.application.services.execution_anti_loss import apply_anti_loss_seed_discord
from src.application.services.execution_signal_skip import metrics_block_execution, parse_signal_skip_config


def _base_metrics(**extra):
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "PUT",
        "resolved_direction": "PUT",
        "exec_direction": "PUT",
        "loss_clf_p_loss": 0.92,
        "loss_clf_auto_learn": False,
        "closed_micro_candle_dir": "CALL",
        "fusion_blocked_tcn_pos_edge": True,
        "kelly_fraction_scale": 1.0,
        "pending_loss_total": 0.0,
    }
    metrics.update(extra)
    metrics.setdefault("ops_window_candle_dir", metrics.get("closed_micro_candle_dir"))
    metrics.setdefault("ops_window_candle_body", metrics.get("closed_micro_candle_body"))
    metrics.setdefault("ops_window_stamped", bool(metrics.get("closed_micro_candle_stamped")))
    return metrics


def test_parse_anti_loss_knobs_from_ssot():
    cfg = parse_signal_skip_config({})
    assert cfg["anti_loss_seed_discord_enabled"] is True
    assert cfg["anti_loss_p_loss_floor"] == pytest.approx(0.85)
    assert cfg["anti_loss_require_seed"] is True
    assert cfg["anti_loss_hard_skip"] is True
    assert cfg["anti_loss_soft_kelly_mult"] == pytest.approx(0.25)
    assert cfg["anti_loss_require_tcn_pos_edge"] is True
    assert cfg["anti_loss_min_candle_body"] == pytest.approx(0.02)
    assert cfg["anti_loss_live_weak_candle_enabled"] is True
    assert cfg["anti_loss_live_confirm_enabled"] is False
    assert cfg["anti_loss_live_confirm_min_body"] == pytest.approx(0.05)
    assert cfg["anti_loss_live_exec_candle_enabled"] is True
    for key, val in (
        ("anti_loss_p_loss_floor", 1.5),
        ("anti_loss_soft_kelly_mult", 0.0),
        ("anti_loss_min_candle_body", -0.1),
        ("anti_loss_live_confirm_min_body", 0.01),
    ):
        with pytest.raises(ValueError, match=key):
            parse_signal_skip_config({key: val})


def test_anti_loss_explore_hard_skip():
    metrics = _base_metrics()
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["gate_reason"] == "anti_loss_seed_discord"
    assert metrics["signal_status"] == "SKIP:ANTI_LOSS_SEED_DISCORD"
    assert metrics["execution_candidate_ready"] is False
    assert metrics_block_execution(metrics) is True


def test_anti_loss_recover_pend_hard_skip():
    metrics = _base_metrics(pending_loss_total=1.5)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["gate_reason"] == "anti_loss_seed_discord"
    assert metrics["execution_candidate_ready"] is False
    assert metrics.get("anti_loss_soft") is None
    assert metrics_block_execution(metrics) is True


def test_anti_loss_candle_agrees_noop():
    metrics = _base_metrics(
        closed_micro_candle_dir="PUT",
        ops_window_candle_dir="PUT",
        closed_micro_candle_body=0.174,
        ops_window_candle_body=0.174,
    )
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_seed_discord") is None


def test_anti_loss_c5_agrees_ignores_prev_bar_lag():
    metrics = _base_metrics()
    metrics.pop("closed_micro_candle_dir")
    metrics["ops_window_candle_dir"] = "PUT"
    metrics["ops_window_candle_body"] = 0.174
    metrics["scale_micro_prev_bar_dir"] = "CALL"
    metrics["scale_micro_bar_dir"] = "PUT"
    metrics["closed_micro_candle_body"] = 0.174
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_seed_discord") is None


def test_anti_loss_c21_weak_body_skips():
    metrics = _base_metrics(closed_micro_candle_dir="PUT", closed_micro_candle_body=0.015)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["anti_loss_why"] == "seed_weak_candle"
    assert metrics["anti_loss_body"] == pytest.approx(0.015)
    assert metrics["gate_reason"] == "anti_loss_seed_discord"


def test_anti_loss_disabled_noop():
    metrics = _base_metrics()
    cfg = parse_signal_skip_config({"anti_loss_seed_discord_enabled": False})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_below_p_loss_floor_noop():
    metrics = _base_metrics(loss_clf_p_loss=0.70)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_requires_tcn_pos_edge():
    metrics = _base_metrics(fusion_blocked_tcn_pos_edge=False, calibrated_prob=0.51, raw_prob=0.51)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    metrics2 = _base_metrics(fusion_blocked_tcn_pos_edge=False, calibrated_prob=0.51, raw_prob=0.51)
    cfg2 = parse_signal_skip_config({"anti_loss_require_tcn_pos_edge": False})
    assert apply_anti_loss_seed_discord(metrics2, cfg=cfg2) is True


def test_anti_loss_force_bypasses():
    metrics = _base_metrics()
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, force=True, cfg=cfg) is False
    assert metrics["execution_candidate_ready"] is True


def test_anti_loss_ready_false_noop():
    metrics = _base_metrics(execution_candidate_ready=False)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False


def test_anti_loss_missing_p_loss_and_bad_side():
    metrics = _base_metrics()
    metrics.pop("loss_clf_p_loss")
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    bad_p = _base_metrics(loss_clf_p_loss="x")
    assert apply_anti_loss_seed_discord(bad_p, cfg=cfg) is False
    bad_side = _base_metrics(tcn_direction="HOLD", resolved_direction="HOLD")
    assert apply_anti_loss_seed_discord(bad_side, cfg=cfg) is False
    no_candle = _base_metrics()
    no_candle.pop("closed_micro_candle_dir", None)
    no_candle.pop("ops_window_candle_dir", None)
    assert apply_anti_loss_seed_discord(no_candle, cfg=cfg) is True
    assert no_candle["anti_loss_why"] == "seed_discord"
    assert no_candle["anti_loss_candle"] == "-"


def test_anti_loss_invalid_body_treated_weak():
    cfg = parse_signal_skip_config({})
    nan_body = _base_metrics(closed_micro_candle_dir="PUT", closed_micro_candle_body=float("nan"))
    assert apply_anti_loss_seed_discord(nan_body, cfg=cfg) is True
    assert nan_body["anti_loss_why"] == "seed_weak_candle"
    neg = _base_metrics(closed_micro_candle_dir="PUT", closed_micro_candle_body=-0.2)
    assert apply_anti_loss_seed_discord(neg, cfg=cfg) is True
    bad = _base_metrics(closed_micro_candle_dir="PUT", closed_micro_candle_body="x")
    assert apply_anti_loss_seed_discord(bad, cfg=cfg) is True
    inf_body = _base_metrics(closed_micro_candle_dir="PUT", closed_micro_candle_body=float("inf"))
    assert apply_anti_loss_seed_discord(inf_body, cfg=cfg) is True


def test_anti_loss_tcn_lock_via_loss_clf_and_fusion_reason():
    cfg = parse_signal_skip_config({})
    via_loss = _base_metrics(fusion_blocked_tcn_pos_edge=False, loss_clf_flip_block_tcn_pos_edge=True)
    assert apply_anti_loss_seed_discord(via_loss, cfg=cfg) is True
    via_reason = _base_metrics(
        fusion_blocked_tcn_pos_edge=False,
        loss_clf_flip_block_tcn_pos_edge=False,
        fusion_reason="tcn_pos_edge",
    )
    assert apply_anti_loss_seed_discord(via_reason, cfg=cfg) is True


def test_anti_loss_soft_when_hard_skip_disabled():
    metrics = _base_metrics(pending_loss_total=2.0)
    cfg = parse_signal_skip_config({"anti_loss_hard_skip": False})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics["anti_loss_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.25)
    assert metrics["execution_candidate_ready"] is True


def test_anti_loss_parse_ssot_default_cfg():
    metrics = _base_metrics()
    assert apply_anti_loss_seed_discord(metrics) is True


def test_anti_loss_live_auto_unstamped_does_not_seed():
    metrics = _base_metrics(loss_clf_auto_learn=True, loss_clf_p_loss=0.96)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_seed_discord") is None


def test_anti_loss_live_exec_discord_strict():
    """Testa veto mandatorio se sinal CALL e vela PUT sem orquestrador."""
    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="PUT",
        ops_window_candle_body=0.001,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["gate_reason"] == "live_exec_discord"
    assert metrics["signal_status"] == "SKIP:LIVE_EXEC_DISCORD"


def test_anti_loss_live_exec_flip_to_candle():
    """Testa inversao inteligente para o lado da vela com orch ativo."""
    from unittest.mock import MagicMock

    orch = MagicMock()
    orch.stream = MagicMock()
    orch.symbols = ["R_10"]
    orch.anchor = "R_10"
    orch.stream.get_mini_numpy_series.return_value = None

    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="PUT",
        ops_window_candle_body=2.5,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is False
    assert metrics["exec_direction"] == "PUT"
    assert metrics["resolved_direction"] == "PUT"
    assert metrics["anti_loss_flipped_to_candle"] is True
    assert metrics["anti_loss_why"] == "live_exec_flip_to_candle"
    assert metrics["anti_loss_soft"] is True


def test_anti_loss_rsi_momentum_veto():
    """Testa veto de CALL para RSI < 0.32 e PUT para RSI > 0.68."""
    metrics_call = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        indicators={"rsi": 0.30},
    )
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics_call, cfg=cfg) is True
    assert metrics_call["gate_reason"] == "anti_loss_rsi_momentum"
    assert metrics_call["signal_status"] == "SKIP:ANTI_LOSS_RSI_MOMENTUM"

    metrics_put = _base_metrics(
        ops_window_stamped=True,
        exec_direction="PUT",
        resolved_direction="PUT",
        ops_window_candle_dir="PUT",
        ops_window_candle_body=0.5,
        indicators={"rsi": 0.70},
    )
    assert apply_anti_loss_seed_discord(metrics_put, cfg=cfg) is True
    assert metrics_put["gate_reason"] == "anti_loss_rsi_momentum"
    assert metrics_put["signal_status"] == "SKIP:ANTI_LOSS_RSI_MOMENTUM"


def test_anti_loss_ema_slope_veto():
    """Testa veto de EMA slope quando EMA21[-1] < EMA21[-3] para CALL."""
    from unittest.mock import MagicMock

    import numpy as np

    stream = MagicMock()
    # Closes descendentes onde EMA21[-1] < EMA21[-3]
    closes = np.linspace(5000, 4800, 30)
    # Colocamos o ultimo close acima da EMA9 temporariamente (repique), mas a EMA21 segue inclinando para baixo
    closes[-1] = closes[-2] + 20.0
    stream.get_mini_numpy_series.return_value = closes
    orch = MagicMock(stream=stream, symbols=["R_10"], anchor="R_10")
    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        indicators={"rsi": 0.50},
    )
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is True
    assert metrics["gate_reason"] in {"anti_loss_ema_slope", "anti_loss_ema_trend"}
