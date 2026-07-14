"""Testes unitarios para recovery_conviction."""

from src.domain.risk.recovery_conviction import (
    recovery_dl_conviction_ok,
    recovery_dl_entry_allowed,
    recovery_min_conviction,
)


def test_recovery_min_conviction_escalates_with_linear_losses():
    cfg = {}
    dlambert = {"recovery_sizing_conviction": 0.58, "recovery_min_conviction": 0.64}
    assert recovery_min_conviction(cfg, dlambert, pending_loss={"RDBEAR": 5.0}, consecutive_losses_linear=4) >= 0.64


def test_recovery_dl_entry_allowed_forced():
    metrics = {"deploy_ok": True, "val_accuracy": 0.1, "trade_score": 0.1}
    assert recovery_dl_entry_allowed(
        metrics,
        {},
        {"recovery_min_val_accuracy": 0.9},
        pending_loss={"RDBEAR": 1.0},
        recovery_forced=True,
    )


def test_recovery_dl_conviction_ok_handles_vetoed_none_scores():
    metrics = {
        "deploy_ok": True,
        "val_accuracy": 0.72,
        "trade_score": None,
        "conviction": None,
        "raw_prob": 0.70,
    }
    assert (
        recovery_dl_conviction_ok(
            metrics,
            {},
            {"recovery_min_val_accuracy": 0.50, "recovery_sizing_conviction": 0.58},
            pending_loss={"RDBEAR": 1.0},
        )
        is True
    )


def test_recovery_dl_conviction_ok_deploy_false():
    assert recovery_dl_conviction_ok({"deploy_ok": False}, {}, {}, pending_loss={}) is False


def test_recovery_min_conviction_force_pending_and_single_loss():
    cfg = {"recovery_force_pending_min": 5.0, "recovery_min_conviction": 0.55}
    dlambert = {"recovery_sizing_conviction": 0.70}
    value = recovery_min_conviction(
        cfg,
        dlambert,
        pending_loss={"RDBEAR": 10.0},
        consecutive_losses_linear=1,
    )
    assert value <= 0.58


def test_recovery_dl_conviction_ok_rejects_low_val():
    metrics = {"deploy_ok": True, "val_accuracy": 0.40, "trade_score": 0.70, "raw_prob": 0.70}
    assert (
        recovery_dl_conviction_ok(
            metrics,
            {},
            {"recovery_min_val_accuracy": 0.50},
            pending_loss={"RDBEAR": 1.0},
        )
        is False
    )


def test_recovery_min_conviction_zero_sizing_reads_force_min():
    cfg = {}
    dlambert = {"recovery_sizing_conviction": 0.0, "recovery_min_conviction": 0.61}
    assert recovery_min_conviction(cfg, dlambert, pending_loss={}, consecutive_losses_linear=0) >= 0.61


def test_recovery_min_conviction_defaults_when_all_zero():
    cfg = {}
    dlambert = {"recovery_sizing_conviction": 0.0, "recovery_min_conviction": 0.0}
    assert recovery_min_conviction(cfg, dlambert, pending_loss={}) == 0.58


def test_recovery_dl_entry_allowed_rejects_deploy_false():
    metrics = {"deploy_ok": False, "val_accuracy": 0.9, "trade_score": 0.9}
    assert (
        recovery_dl_entry_allowed(
            metrics,
            {},
            {},
            pending_loss={"RDBEAR": 1.0},
        )
        is False
    )
