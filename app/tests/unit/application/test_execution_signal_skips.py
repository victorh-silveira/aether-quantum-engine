"""SKIP doji, neg_edge, exec_vs_candle e scale_candle_discord."""

from types import SimpleNamespace

import pytest

from src.application.services.execution_signal_skips import (
    should_skip_acc_floor,
    should_skip_doji,
    should_skip_exec_vs_candle,
    should_skip_neg_edge,
    should_skip_scale_candle_discord,
)
from src.domain.models.trade import TradeDirection


def test_should_skip_acc_floor_below_soft_min():
    metrics = {"val_accuracy": 0.5242}
    orch = SimpleNamespace(config={"deep_learning": {"deploy_gate": {"soft_min_val_accuracy": 0.53}}})
    assert should_skip_acc_floor(metrics, {"skip_below_soft_min_acc": True}, orch=orch) is True
    assert metrics["skip_reason"] == "acc_floor"
    assert metrics["signal_status"] == "SKIP:acc_floor"
    assert metrics["execution_candidate_ready"] is False


def test_should_skip_acc_floor_passes_at_soft_min():
    metrics = {"val_accuracy": 0.53}
    assert (
        should_skip_acc_floor(
            metrics,
            {"skip_below_soft_min_acc": True},
            orch=SimpleNamespace(config={"deep_learning": {"deploy_gate": {"soft_min_val_accuracy": 0.53}}}),
        )
        is False
    )
    assert metrics.get("skip_reason") is None


def test_should_skip_acc_floor_disabled_or_force():
    metrics = {"val_accuracy": 0.50}
    assert should_skip_acc_floor(metrics, {"skip_below_soft_min_acc": False}) is False
    assert should_skip_acc_floor(metrics, {"skip_below_soft_min_acc": True}, force=True) is False


def test_should_skip_doji_when_stamped_without_side():
    metrics = {"closed_micro_candle_stamped": True, "closed_micro_candle_dir": None}
    assert should_skip_doji(metrics, {"skip_doji": True}) is True
    assert metrics["skip_reason"] == "doji"


def test_should_skip_doji_passes_with_call_put():
    metrics = {"closed_micro_candle_stamped": True, "closed_micro_candle_dir": "CALL"}
    assert should_skip_doji(metrics, {"skip_doji": True}) is False


def test_should_skip_neg_edge_when_cal_side_edge_non_positive():
    metrics = {"cal_side_edge": -0.053, "pending_loss_total": 0.0}
    assert should_skip_neg_edge(metrics, {"skip_neg_edge": True}) is True
    assert metrics["skip_reason"] == "neg_edge"
    assert metrics["skip_cal_side_edge"] == pytest.approx(-0.053)


def test_should_skip_neg_edge_waived_when_pend_material():
    metrics = {"cal_side_edge": -0.053, "pending_loss_total": 20.0}
    assert should_skip_neg_edge(metrics, {"skip_neg_edge": True}) is False


def test_should_skip_doji_waived_when_pend_material():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": None,
        "pending_loss_total": 20.0,
    }
    assert should_skip_doji(metrics, {"skip_doji": True}) is False


def test_should_skip_exec_vs_candle_waived_when_pend_material():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "pending_loss_total": 20.0,
    }
    assert (
        should_skip_exec_vs_candle(
            metrics,
            TradeDirection.CALL,
            {"skip_exec_vs_candle": True},
        )
        is False
    )


def test_should_skip_neg_edge_zero_is_skip():
    metrics = {"cal_side_edge": 0.0, "pending_loss_total": 0.0}
    assert should_skip_neg_edge(metrics, {"skip_neg_edge": True}) is True


def test_should_skip_neg_edge_positive_passes():
    metrics = {"cal_side_edge": 0.012}
    assert should_skip_neg_edge(metrics, {"skip_neg_edge": True}) is False


def test_should_skip_neg_edge_below_min_edge_execute():
    metrics = {"cal_side_edge": 0.005, "pending_loss_total": 0.0}
    cfg = {"skip_neg_edge": True, "min_edge_execute": 0.01}
    assert should_skip_neg_edge(metrics, cfg) is True
    assert metrics["skip_reason"] == "neg_edge"


def test_should_skip_exec_vs_candle_disagreement():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
    }
    assert (
        should_skip_exec_vs_candle(
            metrics,
            TradeDirection.CALL,
            {"skip_exec_vs_candle": True},
        )
        is True
    )
    assert metrics["exec_pre_skip"] == "CALL"
    assert metrics["candle_dir"] == "PUT"


def test_should_skip_exec_vs_candle_agreement():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
    }
    assert (
        should_skip_exec_vs_candle(
            metrics,
            TradeDirection.PUT,
            {"skip_exec_vs_candle": True},
        )
        is False
    )


def test_should_skip_scale_rescue_when_adapt_matches_candle_against_tcn():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "exec_direction_pre_scale": "CALL",
        "exec_direction": "PUT",
        "scale_adapted": True,
        "scale_adapt_reason": "explos_vs_tcn",
        "pending_loss_total": 0.0,
    }
    assert should_skip_scale_candle_discord(metrics, {"skip_scale_candle_discord": True}) is True
    assert metrics["skip_reason"] == "scale_rescue"


def test_should_skip_scale_rescue_waived_when_pend_material():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "exec_direction_pre_scale": "CALL",
        "exec_direction": "PUT",
        "scale_adapted": True,
        "scale_adapt_reason": "explos_vs_tcn",
        "pending_loss_total": 20.51,
    }
    assert should_skip_scale_candle_discord(metrics, {"skip_scale_candle_discord": True}) is False
    assert metrics.get("skip_reason") is None


def test_should_skip_scale_candle_conflict_when_candle_overrides_regime():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "CALL",
        "exec_direction_pre_scale": "CALL",
        "exec_direction": "CALL",
        "scale_adapted": True,
        "scale_adapt_reason": "candle_vs_tcn",
        "scale_adapt_from": "PUT",
        "scale_adapt_to": "CALL",
        "pending_loss_total": 0.0,
    }
    assert should_skip_scale_candle_discord(metrics, {"skip_scale_candle_discord": True}) is True
    assert metrics["skip_reason"] == "scale_candle_conflict"


def test_should_skip_scale_candle_discord_clean_alignment_passes():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "CALL",
        "exec_direction_pre_scale": "CALL",
        "exec_direction": "CALL",
        "scale_adapted": False,
        "scale_adapt_reason": "aligned",
        "pending_loss_total": 0.0,
    }
    assert should_skip_scale_candle_discord(metrics, {"skip_scale_candle_discord": True}) is False


def test_should_skip_scale_candle_discord_disabled_or_force():
    metrics = {
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "exec_direction_pre_scale": "CALL",
        "exec_direction": "PUT",
        "scale_adapted": True,
        "scale_adapt_reason": "explos_vs_tcn",
    }
    assert should_skip_scale_candle_discord(metrics, {"skip_scale_candle_discord": False}) is False
    assert should_skip_scale_candle_discord(metrics, {"skip_scale_candle_discord": True}, force=True) is False


def test_signal_skip_helpers_edge_cases():
    from src.application.services.execution_signal_skips import (
        _closed_candle_dir,
        _material_pending_floor,
        _pending_loss_total,
        resolve_soft_min_val_accuracy,
    )

    assert _closed_candle_dir({"closed_micro_candle_stamped": True, "closed_micro_candle_dir": "DOJI"}) is None
    assert _material_pending_floor({"material_pending_min": 1.25}) == pytest.approx(1.25)
    assert _material_pending_floor({"material_pending_min": "bad"}) == pytest.approx(0.5)
    assert _pending_loss_total({"pending_loss_total": "x"}) == pytest.approx(0.0)
    assert resolve_soft_min_val_accuracy(None) == pytest.approx(0.53)
    assert resolve_soft_min_val_accuracy(SimpleNamespace(config={})) == pytest.approx(0.53)


def test_should_skip_acc_floor_missing_or_bad_accuracy():
    assert should_skip_acc_floor({}, {"skip_below_soft_min_acc": True}) is False
    assert should_skip_acc_floor({"val_accuracy": "bad"}, {"skip_below_soft_min_acc": True}) is False


def test_should_skip_doji_without_stamp_passes():
    assert should_skip_doji({"closed_micro_candle_stamped": False}, {"skip_doji": True}) is False


def test_should_skip_neg_edge_falls_back_to_edge_key():
    metrics = {"edge": -0.01, "pending_loss_total": 0.0}
    assert should_skip_neg_edge(metrics, {"skip_neg_edge": True}) is True


def test_should_skip_neg_edge_missing_or_bad_edge_passes():
    assert should_skip_neg_edge({}, {"skip_neg_edge": True}) is False
    assert should_skip_neg_edge({"cal_side_edge": "x"}, {"skip_neg_edge": True}) is False
    assert should_skip_neg_edge({"cal_side_edge": 0.05}, {"skip_neg_edge": True, "min_edge_execute": "bad"}) is False
    assert should_skip_doji({}, {"skip_doji": False}) is False
    assert should_skip_exec_vs_candle({}, TradeDirection.CALL, {"skip_exec_vs_candle": False}) is False


def test_should_skip_exec_vs_candle_without_closed_candle():
    assert (
        should_skip_exec_vs_candle(
            {"closed_micro_candle_stamped": False},
            TradeDirection.CALL,
            {"skip_exec_vs_candle": True},
        )
        is False
    )


def test_should_skip_scale_discord_without_closed_candle():
    metrics = {
        "closed_micro_candle_stamped": False,
        "exec_direction_pre_scale": "CALL",
        "exec_direction": "PUT",
        "scale_adapted": True,
    }
    assert should_skip_scale_candle_discord(metrics, {"skip_scale_candle_discord": True}) is False
