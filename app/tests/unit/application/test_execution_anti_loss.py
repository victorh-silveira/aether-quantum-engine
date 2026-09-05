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
        **extra,
    }
    metrics.setdefault("ops_window_candle_dir", metrics.get("closed_micro_candle_dir"))
    metrics.setdefault("ops_window_candle_body", metrics.get("closed_micro_candle_body"))
    metrics.setdefault("ops_window_stamped", bool(metrics.get("closed_micro_candle_stamped")))
    return metrics


def _cfg(overrides=None, **kwargs):
    base = {
        "anti_loss_seed_discord_enabled": True,
        "anti_loss_live_weak_candle_enabled": True,
        "anti_loss_live_confirm_enabled": True,
        "anti_loss_allow_candle_flip": True,
    }
    if isinstance(overrides, dict):
        base.update(overrides)
    base.update(kwargs)
    return parse_signal_skip_config(base)


def test_parse_anti_loss_knobs_from_ssot():
    cfg = parse_signal_skip_config()
    assert cfg["anti_loss_seed_discord_enabled"] is False and cfg["anti_loss_hard_skip"] is True
    assert cfg["anti_loss_p_loss_floor"] == pytest.approx(0.85)
    assert cfg["anti_loss_require_seed"] is True
    assert float(cfg["anti_loss_soft_kelly_mult"]) == pytest.approx(0.55)
    assert cfg["anti_loss_require_tcn_pos_edge"] is True
    assert cfg["anti_loss_min_candle_body"] == pytest.approx(0.10)
    assert cfg["anti_loss_live_weak_candle_enabled"] is False and cfg["anti_loss_live_confirm_enabled"] is False
    assert (
        cfg["anti_loss_live_confirm_min_body"] == pytest.approx(0.15)
        and cfg["anti_loss_live_exec_candle_enabled"] is False
    )
    assert cfg["anti_loss_rsi_min"] == pytest.approx(0.30) and cfg["anti_loss_rsi_max"] == pytest.approx(0.70)
    for k, v in (
        ("anti_loss_p_loss_floor", 1.5),
        ("anti_loss_soft_kelly_mult", 0.0),
        ("anti_loss_min_candle_body", -0.1),
    ):
        with pytest.raises(ValueError, match=k):
            _cfg({k: v})


def test_anti_loss_explore_hard_skip():
    metrics = _base_metrics()
    cfg = _cfg({"anti_loss_seed_discord_enabled": True, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert (
        metrics["gate_reason"] == "anti_loss_seed_discord" and metrics["signal_status"] == "SKIP:ANTI_LOSS_SEED_DISCORD"
    )
    assert metrics["execution_candidate_ready"] is False and metrics_block_execution(metrics) is True


def test_anti_loss_recover_pend_hard_skip():
    metrics = _base_metrics(pending_loss_total=1.5)
    cfg = _cfg({"anti_loss_hard_skip": True})
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
    cfg = _cfg({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_seed_discord") is None


def test_anti_loss_c5_agrees_ignores_prev_bar_lag():
    metrics = dict(
        _base_metrics(),
        ops_window_candle_dir="PUT",
        ops_window_candle_body=0.174,
        scale_micro_prev_bar_dir="CALL",
        scale_micro_bar_dir="PUT",
        closed_micro_candle_body=0.174,
    )
    metrics.pop("closed_micro_candle_dir", None)
    assert apply_anti_loss_seed_discord(metrics, cfg=_cfg({})) is False
    assert metrics.get("anti_loss_seed_discord") is None


def test_anti_loss_c21_weak_body_skips():
    metrics = _base_metrics(closed_micro_candle_dir="PUT", closed_micro_candle_body=0.015)
    cfg = _cfg({"anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["anti_loss_why"] == "seed_weak_candle" and metrics["anti_loss_body"] == pytest.approx(0.015)
    assert metrics["gate_reason"] == "anti_loss_seed_discord"


def test_anti_loss_disabled_noop():
    assert apply_anti_loss_seed_discord(_base_metrics(), cfg=_cfg({"anti_loss_seed_discord_enabled": False})) is False


def test_anti_loss_below_p_loss_floor_noop():
    assert apply_anti_loss_seed_discord(_base_metrics(loss_clf_p_loss=0.70), cfg=_cfg({})) is False


def test_anti_loss_requires_tcn_pos_edge():
    metrics = _base_metrics(fusion_blocked_tcn_pos_edge=False, calibrated_prob=0.51, raw_prob=0.51)
    assert apply_anti_loss_seed_discord(metrics, cfg=_cfg({})) is False
    metrics2 = _base_metrics(fusion_blocked_tcn_pos_edge=False, calibrated_prob=0.51, raw_prob=0.51)
    cfg2 = _cfg({"anti_loss_require_tcn_pos_edge": False, "anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics2, cfg=cfg2) is True


def test_anti_loss_force_bypasses():
    metrics = _base_metrics()
    cfg = _cfg({})
    assert apply_anti_loss_seed_discord(metrics, force=True, cfg=cfg) is False
    assert metrics["execution_candidate_ready"] is True
    assert apply_anti_loss_seed_discord(_base_metrics(execution_candidate_ready=False), cfg=cfg) is False


def test_anti_loss_missing_p_loss_and_bad_side():
    assert apply_anti_loss_seed_discord(_base_metrics(loss_clf_p_loss=None), cfg=_cfg({})) is False
    assert apply_anti_loss_seed_discord(_base_metrics(loss_clf_p_loss="x"), cfg=_cfg({})) is False
    assert (
        apply_anti_loss_seed_discord(_base_metrics(tcn_direction="HOLD", resolved_direction="HOLD"), cfg=_cfg({}))
        is False
    )
    no_candle = _base_metrics()
    no_candle.pop("closed_micro_candle_dir", None)
    no_candle.pop("ops_window_candle_dir", None)
    assert apply_anti_loss_seed_discord(no_candle, cfg=_cfg({"anti_loss_hard_skip": True})) is True
    assert no_candle["anti_loss_why"] == "seed_discord" and no_candle["anti_loss_candle"] == "-"


def test_anti_loss_invalid_body_treated_weak():
    cfg = _cfg({"anti_loss_hard_skip": True})
    for body_val in (float("nan"), -0.2, "x", float("inf")):
        m = _base_metrics(closed_micro_candle_dir="PUT", closed_micro_candle_body=body_val)
        assert apply_anti_loss_seed_discord(m, cfg=cfg) is True
        assert m["anti_loss_why"] == "seed_weak_candle"


def test_anti_loss_tcn_lock_via_loss_clf_and_fusion_reason():
    cfg = _cfg({"anti_loss_hard_skip": True})
    m1 = _base_metrics(fusion_blocked_tcn_pos_edge=False, loss_clf_flip_block_tcn_pos_edge=True)
    m2 = _base_metrics(
        fusion_blocked_tcn_pos_edge=False, loss_clf_flip_block_tcn_pos_edge=False, fusion_reason="tcn_pos_edge"
    )
    assert apply_anti_loss_seed_discord(m1, cfg=cfg) is True and apply_anti_loss_seed_discord(m2, cfg=cfg) is True


def test_anti_loss_soft_when_hard_skip_disabled():
    metrics = _base_metrics(pending_loss_total=2.0)
    cfg = _cfg({"anti_loss_hard_skip": False, "anti_loss_soft_kelly_mult": 0.25})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics["anti_loss_soft"] is True and metrics["kelly_fraction_scale"] == pytest.approx(0.25)
    assert metrics["execution_candidate_ready"] is True


def test_anti_loss_parse_ssot_default_cfg():
    metrics = _base_metrics()
    assert apply_anti_loss_seed_discord(metrics) is False
    assert metrics.get("execution_candidate_ready") is True


def test_anti_loss_live_auto_unstamped_does_not_seed():
    metrics = _base_metrics(loss_clf_auto_learn=True, loss_clf_p_loss=0.96)
    cfg = _cfg({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_seed_discord") is None


def test_anti_loss_live_exec_discord_strict():
    """Testa veto mandatorio se sinal CALL e vela PUT com flip desativado."""
    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="PUT",
        ops_window_candle_body=0.001,
        indicators={"rsi": 0.50},
    )
    cfg = _cfg(
        {"anti_loss_live_exec_candle_enabled": True, "anti_loss_allow_candle_flip": False, "anti_loss_hard_skip": True}
    )
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["gate_reason"] == "live_exec_discord"
    assert metrics["signal_status"] == "SKIP:LIVE_EXEC_DISCORD"


def test_anti_loss_live_exec_flip_to_candle():
    """Testa inversao inteligente para o lado da vela apenas quando explicitamente habilitado."""
    from unittest.mock import MagicMock

    orch = MagicMock(
        stream=MagicMock(get_mini_numpy_series=MagicMock(return_value=None)), symbols=["R_10"], anchor="R_10"
    )
    orch.config = {
        "deep_learning": {},
        "risk_management": {"params": {"payout_estimate": 0.85}},
        "orchestrator": {"execution": {"signal_skip": {"min_edge_explore": 0.015, "min_edge_recovery": 0.010}}},
    }
    metrics = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="PUT",
        ops_window_candle_body=2.5,
        indicators={"rsi": 0.50},
    )
    cfg_default = _cfg(
        {"anti_loss_live_exec_candle_enabled": True, "anti_loss_allow_candle_flip": False, "anti_loss_hard_skip": True}
    )
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg_default) is True
    assert metrics["gate_reason"] == "live_exec_discord"

    metrics_flip = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="PUT",
        ops_window_candle_body=2.5,
        calibrated_prob=0.38,
        indicators={"rsi": 0.50},
        anti_loss_allow_candle_flip=True,
        fusion_applied=True,
        fusion_p_eff=0.62,
        fusion_p_call=0.62,
        fusion_p_put=0.70,
        fusion_side="CALL",
    )
    cfg_flip = _cfg({"anti_loss_allow_candle_flip": True, "anti_loss_live_exec_candle_enabled": True})
    assert apply_anti_loss_seed_discord(metrics_flip, orch=orch, cfg=cfg_flip) is False
    assert metrics_flip["exec_direction"] == "PUT"
    assert metrics_flip["resolved_direction"] == "PUT"
    assert metrics_flip["anti_loss_flipped_to_candle"] is True
    assert metrics_flip["anti_loss_why"] == "live_exec_flip_to_candle"
    assert metrics_flip["fusion_p_eff"] == pytest.approx(0.70)


def test_anti_loss_rsi_momentum_soft():
    """CALL RSI < 0.30 e PUT RSI > 0.70 soft Kelly; faixa intermediaria passa."""
    metrics_call = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        indicators={"rsi": 0.28},
    )
    cfg = _cfg({"anti_loss_hard_skip": True})
    assert apply_anti_loss_seed_discord(metrics_call, cfg=cfg) is False
    assert metrics_call.get("anti_loss_soft") is True
    assert metrics_call.get("gate_verdict") == "SOFT_SIZE"
    assert metrics_call["anti_loss_why"] == "anti_loss_rsi_momentum"
    assert metrics_call["execution_candidate_ready"] is True
    metrics_mid = _base_metrics(
        ops_window_stamped=True,
        exec_direction="CALL",
        resolved_direction="CALL",
        tcn_direction="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        closed_micro_candle_dir="CALL",
        indicators={"rsi": 0.32},
    )
    assert apply_anti_loss_seed_discord(metrics_mid, cfg=cfg) is False
    assert metrics_mid.get("anti_loss_soft") is None
    assert metrics_mid.get("gate_reason") is None
    metrics_put = _base_metrics(
        ops_window_stamped=True,
        tcn_direction="PUT",
        exec_direction="PUT",
        resolved_direction="PUT",
        closed_micro_candle_dir="PUT",
        ops_window_candle_dir="PUT",
        ops_window_candle_body=0.5,
        indicators={"rsi": 0.72},
    )
    assert apply_anti_loss_seed_discord(metrics_put, cfg=cfg) is False
    assert metrics_put.get("anti_loss_soft") is True
    assert metrics_put.get("gate_verdict") == "SOFT_SIZE"
    assert metrics_put["anti_loss_why"] == "anti_loss_rsi_momentum"
    assert metrics_put["execution_candidate_ready"] is True


def test_anti_loss_rsi_after_candle_flip_keeps_flip_why():
    """Discord candle flipa antes do RSI; RSI extremo no TCN nao gera EMPTY pre-flip."""
    from unittest.mock import MagicMock

    import numpy as np

    stream = MagicMock()
    stream.get_mini_numpy_series.return_value = np.linspace(4800, 5000, 30)
    orch = MagicMock(stream=stream, symbols=["1HZ75V"], anchor="1HZ75V")
    orch.config = {
        "deep_learning": {},
        "risk_management": {"params": {"payout_estimate": 0.85}},
        "orchestrator": {"execution": {"signal_skip": {"min_edge_explore": 0.015, "min_edge_recovery": 0.010}}},
    }
    metrics = _base_metrics(
        ops_window_stamped=True,
        tcn_direction="PUT",
        exec_direction="PUT",
        resolved_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        closed_micro_candle_body=0.5,
        calibrated_prob=0.62,
        indicators={"rsi": 0.85},
    )
    cfg = _cfg(
        {
            "anti_loss_hard_skip": True,
            "anti_loss_allow_candle_flip": True,
            "anti_loss_live_exec_candle_enabled": False,
        }
    )
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg, orch=orch, symbol="1HZ75V") is False
    assert metrics.get("anti_loss_flipped_to_candle") is True
    assert metrics["exec_direction"] == "CALL"
    assert metrics["anti_loss_why"] == "live_exec_flip_to_candle"
    assert metrics.get("anti_loss_rsi_soft") is not True
    assert metrics.get("anti_loss_soft") is True
    assert metrics["execution_candidate_ready"] is True


def test_anti_loss_rsi_soft_flag_when_flip_already_soft():
    """Apos flip, RSI extremo no lado final marca anti_loss_rsi_soft sem apagar flip."""
    from unittest.mock import MagicMock

    import numpy as np

    stream = MagicMock()
    stream.get_mini_numpy_series.return_value = np.linspace(4800, 5000, 30)
    orch = MagicMock(stream=stream, symbols=["1HZ75V"], anchor="1HZ75V")
    orch.config = {
        "deep_learning": {},
        "risk_management": {"params": {"payout_estimate": 0.85}},
        "orchestrator": {"execution": {"signal_skip": {"min_edge_explore": 0.015, "min_edge_recovery": 0.010}}},
    }
    metrics = _base_metrics(
        ops_window_stamped=True,
        tcn_direction="PUT",
        exec_direction="PUT",
        resolved_direction="PUT",
        closed_micro_candle_dir="CALL",
        ops_window_candle_dir="CALL",
        ops_window_candle_body=0.5,
        closed_micro_candle_body=0.5,
        calibrated_prob=0.62,
        indicators={"rsi": 0.20},
    )
    cfg = _cfg(
        {
            "anti_loss_hard_skip": True,
            "anti_loss_allow_candle_flip": True,
            "anti_loss_live_exec_candle_enabled": False,
        }
    )
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg, orch=orch, symbol="1HZ75V") is False
    assert metrics.get("anti_loss_flipped_to_candle") is True
    assert metrics["exec_direction"] == "CALL"
    assert metrics["anti_loss_why"] == "live_exec_flip_to_candle"
    assert metrics.get("anti_loss_rsi_soft") is True
    assert metrics.get("anti_loss_soft") is True
