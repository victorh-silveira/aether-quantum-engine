"""Testes do playbook senior: RSI align, ADX, Hurst noise, cal floor."""

from __future__ import annotations

from src.application.services.execution_direction_checks import initial_direction_checks
from src.application.services.execution_direction_discordance import apply_technical_agreement
from src.application.services.execution_quality_reject import reject_on_quality_gate
from src.application.services.execution_senior_skip import SENIOR_SKIP_REASONS, is_senior_skip_reason
from src.domain.models.trade import TradeDirection


def _entry(prob: float = 0.62, **extra) -> dict:
    metrics = {
        "deploy_ok": True,
        "execute": True,
        "raw_prob": prob,
        "calibrated_prob": prob,
        "val_accuracy": 0.70,
        "indicators": {"rsi": 0.30, "adx": 0.25, "hurst": 0.60, "di_diff": -0.10},
        **extra,
    }
    return {"direction": None, "metrics": metrics}


def test_senior_skip_catalog_contains_core_reasons():
    assert is_senior_skip_reason("cal_margin_floor")
    assert is_senior_skip_reason("rsi_trend_misalign")
    assert is_senior_skip_reason("hurst_noise")
    assert is_senior_skip_reason("  adx_min  ")
    assert not is_senior_skip_reason("unknown")
    assert not is_senior_skip_reason(None)
    assert "adx_min" in SENIOR_SKIP_REASONS


def test_hard_cal_floor_rejects_cal_054():
    metrics = {"calibrated_prob": 0.54, "raw_prob": 0.54, "val_accuracy": 0.70}
    assert (
        reject_on_quality_gate(
            {},
            metrics,
            dict(metrics),
            {"hard_cal_margin_floor": 0.05, "quality_gate": {"regular": {"min_direction_margin": 0.0}}},
        )
        is True
    )
    assert metrics["gate_reason"] == "cal_margin_floor"


def test_align_rsi_trend_blocks_misalign():
    metrics = {
        "indicators": {"rsi": 0.72, "di_diff": 0.12, "adx": 0.30},
        "direction_margin": 0.04,
        "calibrated_prob": 0.54,
    }
    _prob, veto = apply_technical_agreement(
        metrics,
        TradeDirection.PUT,
        0.30,
        {
            "align_rsi_trend": True,
            "discordance_veto_enabled": True,
            "discordance": {
                "rsi_bias_min": 0.10,
                "rsi_solo_bias_min": 0.14,
                "di_abs_min": 0.05,
                "waiver_margin": 0.10,
                "meta_edge_waiver": 0.50,
            },
        },
    )
    assert veto is True
    assert metrics.get("gate_reason") == "rsi_trend_misalign"


def test_adx_min_skips_weak_trend():
    entry = _entry(0.70)
    entry["metrics"]["indicators"]["adx"] = 0.15
    entry["metrics"]["indicators"]["hurst"] = 0.60
    orch = type("O", (), {"config": {"deep_learning": {"indicator_gating": {"enabled": True, "adx_min": 0.16}}}})()
    result = initial_direction_checks(entry, {}, orch=orch)
    assert result is None
    assert entry["metrics"].get("gate_reason") == "adx_min"


def test_hurst_noise_skips_mid_band():
    entry = _entry(0.70)
    entry["metrics"]["indicators"]["adx"] = 0.30
    entry["metrics"]["indicators"]["hurst"] = 0.50
    orch = type(
        "O",
        (),
        {
            "config": {
                "deep_learning": {
                    "indicator_gating": {
                        "enabled": True,
                        "adx_min": 0.0,
                        "veto_on_noise": True,
                        "noise_hurst_lo": 0.47,
                        "noise_hurst_hi": 0.53,
                        "veto_missing_hurst": True,
                    }
                }
            }
        },
    )()
    result = initial_direction_checks(entry, {}, orch=orch)
    assert result is None
    assert entry["metrics"].get("gate_reason") == "hurst_noise"


def test_hurst_missing_skips_when_configured():
    entry = _entry(0.70)
    entry["metrics"]["indicators"] = {"rsi": 0.55, "adx": 0.30, "di_diff": 0.02}
    orch = type(
        "O",
        (),
        {
            "config": {
                "deep_learning": {
                    "indicator_gating": {
                        "enabled": True,
                        "adx_min": 0.0,
                        "veto_on_noise": True,
                        "noise_hurst_lo": 0.47,
                        "noise_hurst_hi": 0.53,
                        "veto_missing_hurst": True,
                    }
                }
            }
        },
    )()
    result = initial_direction_checks(entry, {}, orch=orch)
    assert result is None
    assert entry["metrics"].get("gate_reason") == "hurst_missing"
