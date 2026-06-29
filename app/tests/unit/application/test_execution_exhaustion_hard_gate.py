"""Testes do hard gate de exaustao RSI+CMO+Keltner."""

from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_exhaustion_hard_gate import (
    adx_super_trend_exempt,
    dl_weight_retention,
    hard_gate_score_penalty,
    severe_buy_exhaustion,
    severe_sell_exhaustion,
)
from src.domain.models.trade import TradeDirection


_CFG = {
    "exhaustion_gate": {
        "hard_gate_enabled": True,
        "rsi_overbought": 0.73,
        "rsi_oversold": 0.27,
        "cmo_bull": 0.48,
        "cmo_bear": -0.48,
        "keltner_overbought": 1.15,
        "keltner_oversold": -0.15,
        "dl_weight_retention": 0.20,
        "adx_super_trend_min": 0.40,
        "hard_gate_score_penalty": 0.25,
        "min_penalty_skip": 0.12,
    }
}


def test_severe_buy_exhaustion_triple_condition():
    metrics = {"indicators": {"rsi": 0.74, "cmo": 0.50, "keltner": 1.20}}
    assert severe_buy_exhaustion(metrics, cfg=_CFG) is True


def test_severe_buy_exhaustion_missing_keltner():
    metrics = {"indicators": {"rsi": 0.74, "cmo": 0.50, "keltner": 1.10}}
    assert severe_buy_exhaustion(metrics, cfg=_CFG) is False


def test_dl_weight_retention_attenuates_call():
    metrics = {"indicators": {"rsi": 0.74, "cmo": 0.50, "keltner": 1.20, "adx": 0.25}}
    retention = dl_weight_retention(metrics, TradeDirection.CALL, cfg=_CFG)
    assert retention == 0.20


def test_dl_weight_retention_exempt_on_super_trend():
    metrics = {"indicators": {"rsi": 0.74, "cmo": 0.50, "keltner": 1.20, "adx": 0.45}}
    retention = dl_weight_retention(metrics, TradeDirection.CALL, cfg=_CFG)
    assert retention == 1.0
    assert adx_super_trend_exempt(metrics, cfg=_CFG) is True


def test_severe_sell_exhaustion():
    metrics = {"indicators": {"rsi": 0.20, "cmo": -0.55, "keltner": -0.20}}
    assert severe_sell_exhaustion(metrics, cfg=_CFG) is True


def test_hard_gate_score_penalty_above_skip_floor():
    assert hard_gate_score_penalty(cfg=_CFG) >= 0.37


def test_hard_gate_disabled():
    metrics = {"indicators": {"rsi": 0.74, "cmo": 0.50, "keltner": 1.20}}
    disabled = {"exhaustion_gate": {"hard_gate_enabled": False}}
    assert severe_buy_exhaustion(metrics, cfg=disabled) is False
    assert severe_sell_exhaustion(metrics, cfg=disabled) is False
    assert dl_weight_retention(metrics, TradeDirection.CALL, cfg=disabled) == 1.0


def test_dl_weight_retention_attenuates_put_on_sell_exhaustion():
    metrics = {"indicators": {"rsi": 0.20, "cmo": -0.55, "keltner": -0.20, "adx": 0.25}}
    retention = dl_weight_retention(metrics, TradeDirection.PUT, cfg=_CFG)
    assert retention == 0.20


def test_resolve_hard_gate_sets_retention_and_penalty():
    exec_cfg = {
        "exhaustion_gate": {
            "hard_gate_enabled": True,
            "rsi_overbought": 0.73,
            "cmo_bull": 0.48,
            "keltner_overbought": 1.15,
            "dl_weight_retention": 0.20,
            "adx_super_trend_min": 0.40,
            "hard_gate_score_penalty": 0.25,
            "min_penalty_skip": 0.12,
        }
    }
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.72,
            "val_accuracy": 0.62,
            "indicators": {
                "rsi": 0.74,
                "cmo": 0.50,
                "keltner": 1.20,
                "adx": 0.25,
                "hurst": 0.55,
                "vol_ratio": 1.0,
            },
        },
    }
    result = resolve_execution_direction(entry, exec_cfg=exec_cfg)
    assert result is not None
    _, metrics = result
    assert metrics.get("exhaustion_hard_gate") is True
    assert metrics.get("exhaustion_dl_retention") == 0.20
    assert float(metrics.get("exhaustion_penalty", 0.0)) >= 0.37
