from src.domain.risk.martingale_gate import (
    _martingale_raw_block,
    apply_win_to_pending_loss,
    martingale_allowed,
    martingale_block_reason,
    martingale_dl_metrics_block,
    martingale_repeat_loss_blocked,
)


def test_martingale_dl_metrics_block_gate_and_brier():
    assert martingale_dl_metrics_block({"gate_reason": "deploy"}, max_val_brier=0.28)
    assert not martingale_dl_metrics_block(
        {"gate_reason": "raw_conviction", "deploy_ok": False, "val_brier": 0.2},
        max_val_brier=0.28,
        recovery_pending=True,
    )
    assert martingale_dl_metrics_block({"val_brier": 0.3}, max_val_brier=0.28)
    assert not martingale_dl_metrics_block(
        {"val_brier": 0.3, "gate_reason": "brier"}, max_val_brier=0.28, recovery_pending=True
    )
    assert martingale_dl_metrics_block({"deploy_ok": False, "val_brier": 0.1}, max_val_brier=0.28)
    assert not martingale_dl_metrics_block(
        {"deploy_ok": False, "val_brier": 0.1}, max_val_brier=0.28, recovery_pending=True
    )
    assert not martingale_dl_metrics_block(
        {"deploy_ok": False, "val_brier": 0.1},
        max_val_brier=0.28,
        recovery_pending=True,
    )


def test_martingale_allowed_with_pending_loss_ignores_low_conviction_when_forced():
    assert martingale_allowed(
        pending_loss={"RDBULL": 50.0},
        recovery_threshold=0.72,
        recovery_martingale_min_conviction=0.45,
        conviction=0.40,
        symbol="RDBEAR",
        dl_metrics={"raw_prob": 0.55},
        force_on_pending_loss=True,
    )


def test_martingale_block_reason_raw_conviction_when_forced():
    assert (
        martingale_block_reason(
            pending_loss={"RDBULL": 50.0},
            recovery_threshold=0.72,
            conviction=0.40,
            symbol="RDBEAR",
            dl_metrics={"raw_prob": 0.51},
            recovery_martingale_min_raw=0.52,
            force_on_pending_loss=True,
        )
        == "raw_conviction"
    )


def test_martingale_raw_block_allows_high_raw_side():
    assert _martingale_raw_block(0.40, {"raw_prob": 0.55}, 0.52) is None


def test_martingale_raw_block_skips_when_min_raw_zero():
    assert _martingale_raw_block(0.40, None, 0.0) is None


def test_martingale_raw_block_uses_conviction_without_dl_metrics():
    assert _martingale_raw_block(0.40, None, 0.52) == "raw_conviction"


def test_martingale_allowed_respects_conviction_when_force_disabled():
    assert not martingale_allowed(
        pending_loss={"RDBULL": 50.0},
        recovery_threshold=0.72,
        recovery_martingale_min_conviction=0.45,
        conviction=0.40,
        symbol="RDBEAR",
        force_on_pending_loss=False,
    )
    assert (
        martingale_block_reason(
            pending_loss={"RDBULL": 50.0},
            recovery_threshold=0.72,
            recovery_martingale_min_conviction=0.45,
            conviction=0.40,
            symbol="RDBEAR",
            force_on_pending_loss=False,
        )
        == "conviction"
    )
    assert (
        martingale_block_reason(
            pending_loss={"RDBULL": 50.0},
            recovery_threshold=0.72,
            recovery_martingale_min_conviction=0.40,
            conviction=0.50,
            symbol="RDBEAR",
            force_on_pending_loss=False,
        )
        == "recovery_threshold"
    )


def test_martingale_block_reason_dl_metrics_when_force_disabled():
    assert (
        martingale_block_reason(
            pending_loss={"RDBULL": 50.0},
            recovery_threshold=0.72,
            conviction=0.75,
            symbol="RDBEAR",
            dl_metrics={"gate_reason": "deploy"},
            force_on_pending_loss=False,
        )
        == "dl_metrics"
    )


def test_martingale_block_reason_repeat_loss():
    assert (
        martingale_block_reason(
            pending_loss={"RDBULL": 99.0},
            recovery_threshold=0.72,
            conviction=0.47,
            symbol="RDBULL",
            order_direction="PUT",
            last_loss_symbol="RDBULL",
            last_loss_direction="PUT",
        )
        == "repeat_loss"
    )


def test_martingale_allowed_with_high_brier_when_recovery_pending():
    assert martingale_allowed(
        pending_loss={"RDBULL": 100.0},
        recovery_threshold=0.72,
        conviction=0.44,
        symbol="RDBULL",
        dl_metrics={"gate_reason": "brier", "val_brier": 0.35, "deploy_ok": False},
        order_direction="PUT",
    )


def test_martingale_allowed_after_loss_with_low_conviction_and_gate_reason():
    pending = {"RDBULL": 99.0}
    assert martingale_allowed(
        pending_loss=pending,
        recovery_threshold=0.72,
        recovery_martingale_min_conviction=0.45,
        conviction=0.44,
        symbol="RDBEAR",
        dl_metrics={"gate_reason": "raw_conviction", "val_brier": 0.25, "deploy_ok": False},
        order_direction="PUT",
        last_loss_symbol="RDBULL",
        last_loss_direction="CALL",
    )


def test_martingale_repeat_loss_blocked():
    assert martingale_repeat_loss_blocked("X", "CALL", "X", "CALL")
    assert not martingale_repeat_loss_blocked("X", "PUT", "X", "CALL")


def test_martingale_allowed_and_apply_win():
    assert not martingale_allowed(pending_loss={}, recovery_threshold=0.5, conviction=0.8, symbol="X")
    pending = {"A": 10.0}
    apply_win_to_pending_loss(pending, 4.0)
    assert pending["A"] == 6.0
    apply_win_to_pending_loss(pending, 20.0)
    assert pending["A"] == 0.0
