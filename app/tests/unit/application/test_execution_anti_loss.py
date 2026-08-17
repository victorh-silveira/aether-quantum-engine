"""Gate anti-loss seed + vela discordante do TCN."""

from __future__ import annotations

import pytest

from src.application.services.execution_anti_loss import (
    apply_anti_loss_seed_discord,
    evaluate_anti_loss_seed_discord,
)
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
    return metrics


def test_parse_anti_loss_knobs_from_ssot():
    cfg = parse_signal_skip_config({})
    assert cfg["anti_loss_seed_discord_enabled"] is True
    assert cfg["anti_loss_p_loss_floor"] == pytest.approx(0.85)
    assert cfg["anti_loss_require_seed"] is True
    assert cfg["anti_loss_hard_skip_explore"] is True
    assert cfg["anti_loss_recover_soft_kelly_mult"] == pytest.approx(0.25)
    assert cfg["anti_loss_require_tcn_pos_edge"] is True
    with pytest.raises(ValueError, match="anti_loss_p_loss_floor"):
        parse_signal_skip_config({"anti_loss_p_loss_floor": 1.5})
    with pytest.raises(ValueError, match="anti_loss_recover_soft_kelly_mult"):
        parse_signal_skip_config({"anti_loss_recover_soft_kelly_mult": 0.0})


def test_anti_loss_explore_hard_skip():
    metrics = _base_metrics()
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["gate_reason"] == "anti_loss_seed_discord"
    assert metrics["signal_status"] == "SKIP:ANTI_LOSS_SEED_DISCORD"
    assert metrics["execution_candidate_ready"] is False
    assert metrics_block_execution(metrics) is True


def test_anti_loss_recover_soft_only():
    metrics = _base_metrics(pending_loss_total=1.5)
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("gate_reason") is None
    assert metrics["execution_candidate_ready"] is True
    assert metrics["anti_loss_soft"] is True
    assert metrics["kelly_fraction_scale"] == pytest.approx(0.25)
    assert metrics_block_execution(metrics) is False


def test_anti_loss_live_auto_learn_noop():
    metrics = _base_metrics(loss_clf_auto_learn=True)
    cfg = parse_signal_skip_config({})
    decision = evaluate_anti_loss_seed_discord(metrics, cfg=cfg)
    assert decision["active"] is False
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_seed_discord") is None


def test_anti_loss_candle_agrees_noop():
    metrics = _base_metrics(closed_micro_candle_dir="PUT")
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_seed_discord") is None


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
    assert apply_anti_loss_seed_discord(no_candle, cfg=cfg) is False


def test_anti_loss_pending_orch_none_without_keys():
    cfg = parse_signal_skip_config({})
    metrics = _base_metrics()
    for key in ("pending_loss_total", "pending_total", "recovery_pending"):
        metrics.pop(key, None)
    assert apply_anti_loss_seed_discord(metrics, orch=None, cfg=cfg) is True


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
    metrics = _base_metrics()
    cfg = parse_signal_skip_config({"anti_loss_hard_skip_explore": False})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics["anti_loss_soft"] is True


def test_anti_loss_pending_from_orch_paths():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    cfg = parse_signal_skip_config({})
    metrics = _base_metrics()
    metrics["pending_loss_total"] = "bad"
    orch = SimpleNamespace(risk_manager=None)
    assert apply_anti_loss_seed_discord(metrics, orch=orch, cfg=cfg) is True

    metrics2 = _base_metrics()
    metrics2.pop("pending_loss_total", None)
    orch2 = SimpleNamespace(risk_manager=SimpleNamespace(pending_loss_total=MagicMock(side_effect=ValueError("x"))))
    assert apply_anti_loss_seed_discord(metrics2, orch=orch2, cfg=cfg) is True

    metrics3 = _base_metrics()
    for key in ("pending_loss_total", "pending_total", "recovery_pending"):
        metrics3.pop(key, None)

    class RmMap:
        pending_loss = {"a": 2.0}

    orch3 = SimpleNamespace(risk_manager=RmMap())
    assert apply_anti_loss_seed_discord(metrics3, orch=orch3, cfg=cfg) is False
    assert metrics3["anti_loss_soft"] is True

    metrics4 = _base_metrics()
    for key in ("pending_loss_total", "pending_total", "recovery_pending"):
        metrics4.pop(key, None)

    class RmBadMap:
        pending_loss = {"a": "x"}

    orch4 = SimpleNamespace(risk_manager=RmBadMap())
    assert apply_anti_loss_seed_discord(metrics4, orch=orch4, cfg=cfg) is True

    metrics5 = _base_metrics()
    for key in ("pending_loss_total", "pending_total", "recovery_pending"):
        metrics5.pop(key, None)

    class RmNotMap:
        pending_loss = "nope"

    orch5 = SimpleNamespace(risk_manager=RmNotMap())
    assert apply_anti_loss_seed_discord(metrics5, orch=orch5, cfg=cfg) is True

    metrics6 = _base_metrics(pending_total=3.0)
    metrics6.pop("pending_loss_total", None)
    assert apply_anti_loss_seed_discord(metrics6, cfg=cfg) is False
    assert metrics6["anti_loss_soft"] is True

    metrics7 = _base_metrics()
    for key in ("pending_loss_total", "pending_total", "recovery_pending"):
        metrics7.pop(key, None)

    class RmFn:
        @staticmethod
        def pending_loss_total():
            return 4.0

    orch7 = SimpleNamespace(risk_manager=RmFn())
    assert apply_anti_loss_seed_discord(metrics7, orch=orch7, cfg=cfg) is False

    metrics8 = _base_metrics()
    assert apply_anti_loss_seed_discord(metrics8) is True
