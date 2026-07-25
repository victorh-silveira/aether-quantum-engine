from types import SimpleNamespace

from src.application.services.execution_direction_checks import (
    initial_direction_checks,
    sniper_cfg,
)
from src.domain.models.trade import TradeDirection


def test_sniper_cfg_reads_indicator_gating_from_orch():
    orch = SimpleNamespace(
        config={
            "deep_learning": {
                "indicator_gating": {
                    "enabled": True,
                    "veto_on_noise": True,
                    "noise_hurst_lo": 0.45,
                    "noise_hurst_hi": 0.55,
                    "strong_trend_min": 0.65,
                    "strong_revert_max": 0.35,
                }
            }
        }
    )
    squeeze_cfg, gating_cfg = sniper_cfg({"bb_width_adaptive_squeeze": {"enabled": False}}, orch)
    assert squeeze_cfg["enabled"] is False
    assert "anomaly_ratio" in squeeze_cfg
    assert gating_cfg["enabled"] is True
    assert gating_cfg["veto_on_noise"] is True


def test_initial_direction_checks_rejects_hurst_noise_via_orch():
    orch = SimpleNamespace(
        config={
            "deep_learning": {
                "indicator_gating": {
                    "enabled": True,
                    "veto_on_noise": True,
                    "noise_hurst_lo": 0.45,
                    "noise_hurst_hi": 0.55,
                    "strong_trend_min": 0.65,
                    "strong_revert_max": 0.35,
                }
            }
        }
    )
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.72,
            "indicators": {"hurst": 0.50},
            "execute": True,
        },
    }
    assert initial_direction_checks(entry, {}, orch=orch) is not None
    assert entry["metrics"].get("gate_reason") != "hurst_noise_veto"


def test_initial_direction_checks_skips_neutral_signals():
    entry = {
        "direction": None,
        "metrics": {
            "calibration_mode": "neutral_clamp",
            "calibrated_prob": 0.50,
            "raw_prob": 0.50,
        },
    }
    result = initial_direction_checks(entry, {})
    assert result is None
    assert entry["metrics"]["signal_status"] == "SKIP"
    assert entry["metrics"]["execute"] is False
    assert entry["execute"] is False
    assert entry["metrics"]["regime_skip_cycle"] is True


def test_initial_direction_checks_skips_choppiness_high_noise():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "choppiness_index": 68.5,
            "calibrated_prob": 0.65,
            "raw_prob": 0.65,
            "execute": True,
        },
    }
    result = initial_direction_checks(entry, {})
    assert result is None
    assert entry["metrics"]["signal_status"] == "SKIP"
    assert entry["metrics"]["execute"] is False
    assert entry["execute"] is False
