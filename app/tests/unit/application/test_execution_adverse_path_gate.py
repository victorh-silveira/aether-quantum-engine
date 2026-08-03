from src.application.services.direction_loss_tracker import (
    get_direction_loss_tracker,
    reset_direction_persistence_tracker,
)
from src.application.services.execution_adverse_path_gate import apply_adverse_micro_path_gate
from src.application.services.execution_direction_checks import initial_direction_checks
from src.application.services.execution_runtime_config import resolve_direction_persistence_config
from src.domain.models.trade import TradeDirection


def _exec_cfg(**overrides):
    base = {
        "adverse_path": {
            "enabled": True,
            "rsi_bias_min": 0.08,
            "rsi_hard_bias": 0.12,
            "waiver_margin": 0.40,
        },
        "discordance_veto_enabled": False,
        "price_zone": {"enabled": False},
    }
    base.update(overrides)
    return base


def test_adverse_path_blocks_put_against_call_rsi():
    metrics = {
        "candle_color_direction": "CALL",
        "direction_margin": 0.15,
        "calibrated_prob": 0.35,
        "macro_indicators": {"rsi": 0.62},
    }
    assert apply_adverse_micro_path_gate(metrics, TradeDirection.PUT, _exec_cfg(), skipped_cycles_counter=0) is True
    assert metrics.get("gate_reason") == "adverse_micro_path"
    assert metrics.get("adverse_micro_path_hard") is True
    assert metrics.get("adverse_micro_path_candle") == "CALL"


def test_adverse_path_weak_tcn_defers():
    metrics = {
        "candle_color_direction": "CALL",
        "direction_margin": 0.02,
        "calibrated_prob": 0.52,
        "macro_indicators": {"rsi": 0.65},
    }
    assert apply_adverse_micro_path_gate(metrics, TradeDirection.PUT, _exec_cfg(), skipped_cycles_counter=0) is False
    assert metrics.get("adverse_micro_path_weak_tcn_defer") is True
    assert metrics.get("gate_reason") is None


def test_adverse_path_blocks_even_with_tcn_margin_when_rsi_hard():
    metrics = {
        "candle_color_direction": "CALL",
        "direction_margin": 0.26,
        "calibrated_prob": 0.24,
        "macro_indicators": {"rsi": 0.65},
    }
    assert apply_adverse_micro_path_gate(metrics, TradeDirection.PUT, _exec_cfg(), skipped_cycles_counter=0) is True
    assert metrics.get("gate_reason") == "adverse_micro_path"
    assert metrics.get("adverse_micro_path_margin_waiver") is not True


def test_adverse_path_soft_margin_waiver():
    metrics = {
        "direction_margin": 0.42,
        "calibrated_prob": 0.08,
        "macro_indicators": {"rsi": 0.59},
    }
    assert apply_adverse_micro_path_gate(metrics, TradeDirection.PUT, _exec_cfg(), skipped_cycles_counter=0) is False
    assert metrics.get("adverse_micro_path_margin_waiver") is True
    assert metrics.get("gate_reason") is None


def test_adverse_path_starvation_waiver_soft_only():
    metrics = {
        "direction_margin": 0.13,
        "calibrated_prob": 0.37,
        "macro_indicators": {"rsi": 0.41},
    }
    assert apply_adverse_micro_path_gate(metrics, TradeDirection.CALL, _exec_cfg(), skipped_cycles_counter=6) is False
    assert metrics.get("adverse_micro_path_starvation_waiver") is True


def test_adverse_path_disabled_or_aligned_passes():
    metrics = {
        "candle_color_direction": "PUT",
        "macro_indicators": {"rsi": 0.35},
        "direction_margin": 0.15,
        "calibrated_prob": 0.35,
    }
    assert apply_adverse_micro_path_gate(metrics, TradeDirection.PUT, _exec_cfg(), skipped_cycles_counter=0) is False
    assert (
        apply_adverse_micro_path_gate(
            metrics,
            TradeDirection.CALL,
            {"adverse_path": {"enabled": False}},
            skipped_cycles_counter=0,
        )
        is False
    )
    weak = {"macro_indicators": {"rsi": 0.55}, "direction_margin": 0.15, "calibrated_prob": 0.35}
    assert apply_adverse_micro_path_gate(weak, TradeDirection.PUT, _exec_cfg(), skipped_cycles_counter=0) is False
    no_rsi = {"candle_color_direction": "CALL", "direction_margin": 0.15, "calibrated_prob": 0.35}
    assert apply_adverse_micro_path_gate(no_rsi, TradeDirection.PUT, _exec_cfg()) is False
    rsi_opposes_candle = {
        "candle_color_direction": "PUT",
        "macro_indicators": {"rsi": 0.65},
        "direction_margin": 0.15,
        "calibrated_prob": 0.35,
    }
    assert apply_adverse_micro_path_gate(rsi_opposes_candle, TradeDirection.PUT, _exec_cfg()) is False


def test_initial_direction_checks_adverse_micro_path():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "calibrated_prob": 0.35,
            "direction_margin": 0.15,
            "deploy_ok": True,
            "execute": True,
            "macro_indicators": {"rsi": 0.65, "open": 100.0, "close": 101.5},
        },
    }
    assert initial_direction_checks(entry, _exec_cfg()) is None
    assert entry["metrics"].get("gate_reason") == "adverse_micro_path"


def test_anti_trend_lock_threshold_is_two():
    cfg = resolve_direction_persistence_config({"direction_persistence": {"same_direction_count_threshold": 2}})
    assert cfg["same_direction_count_threshold"] == 2
    reset_direction_persistence_tracker()
    tracker = get_direction_loss_tracker()
    tracker.record_outcome("R_10", "PUT", won=False)
    tracker.record_outcome("R_10", "PUT", won=False)
    assert tracker.anti_trend_lock_active("R_10", TradeDirection.PUT) is True
    reset_direction_persistence_tracker()
