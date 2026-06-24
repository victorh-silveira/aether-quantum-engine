from src.application.services.execution_direction import _entry_gate_blocked
from src.application.services.execution_market_rank import (
    mandatory_pool_eligible,
    market_decision_score,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_uses_entry_direction():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.80, "execute": True},
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_infers_from_raw_prob():
    entry = {
        "direction": None,
        "metrics": {"raw_prob": 0.20},
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_mean_reversion_inversion_put_to_call():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.35,
            "indicators": {
                "hurst": 0.45,
                "adx": 0.22,
                "vol_ratio": 0.80,
                "rsi": 0.38,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL
    assert entry["metrics"]["direction_inverted"] is True


def test_resolve_market_direction_mean_reversion_inversion_call_to_put():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.65,
            "indicators": {
                "hurst": 0.45,
                "adx": 0.22,
                "vol_ratio": 0.80,
                "rsi": 0.62,
            },
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT
    assert entry["metrics"]["direction_inverted"] is True


def test_resolve_market_direction_trend_exhaustion_put_ignored():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.80,
            "execute": False,
            "trend_direction": "PUT",
            "indicators": {
                "vol_ratio": 1.44,
                "adx": 0.15,
                "rsi": 0.44,
                "keltner": 0.28,
            },
        },
    }
    # A tendência seria PUT, mas como está oversold (rsi < 0.45, keltner < 0.30), ignora e retorna dl_dir (CALL)
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_trend_exhaustion_call_ignored():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.20,
            "execute": False,
            "trend_direction": "CALL",
            "indicators": {
                "vol_ratio": 1.44,
                "adx": 0.15,
                "rsi": 0.58,
                "keltner": 0.75,
            },
        },
    }
    # A tendência seria CALL, mas como está overbought (rsi > 0.55, keltner > 0.70), ignora e retorna dl_dir (PUT)
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_resolve_market_direction_unreliable_accuracy_no_inversion():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.49,
            "raw_prob": 0.80,
            "execute": True,
        },
    }
    # Acurácia de 0.49 está abaixo de 0.50, portanto DEVE inverter (inverte CALL para PUT)
    assert resolve_market_direction(entry) == TradeDirection.PUT
    assert entry["metrics"].get("direction_inverted") is True


def test_mandatory_pool_eligible_grey_zone():
    entry_grey = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.49, "raw_prob": 0.80, "execute": False},
    }
    entry_good = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.55, "raw_prob": 0.80, "execute": True},
    }
    entry_inverted = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.42, "raw_prob": 0.80, "execute": False},
    }
    assert mandatory_pool_eligible(entry_grey) is True
    assert mandatory_pool_eligible(entry_good) is True
    assert mandatory_pool_eligible(entry_inverted) is True


def test_entry_gate_blocked_grey_zone():
    metrics_grey = {"val_accuracy": 0.49, "raw_prob": 0.80, "execute": False}
    metrics_good = {"val_accuracy": 0.55, "raw_prob": 0.80, "execute": True}
    metrics_inverted = {"val_accuracy": 0.42, "raw_prob": 0.80, "execute": False}
    assert _entry_gate_blocked(metrics_grey) is False
    assert _entry_gate_blocked(metrics_good) is False
    assert _entry_gate_blocked(metrics_inverted) is False


def test_resolve_market_direction_mean_reversion_disabled():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.35,
            "indicators": {
                "hurst": 0.45,
                "adx": 0.22,
                "vol_ratio": 0.80,
                "rsi": 0.38,
            },
        },
    }
    # With mean reversion disabled, it should not invert PUT to CALL
    assert resolve_market_direction(entry, mean_reversion_enabled=False) == TradeDirection.PUT
    assert entry["metrics"].get("direction_inverted") is not True


def test_resolve_market_direction_low_accuracy_disabled():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.42,
            "raw_prob": 0.80,
            "trend_direction": "PUT",
            "indicators": {
                "rsi": 0.50,
                "keltner": 0.50,
            },
        },
    }
    # With low accuracy inversion disabled, it should not invert CALL to PUT
    assert resolve_market_direction(entry, low_accuracy_enabled=False) == TradeDirection.CALL
    assert entry["metrics"].get("direction_inverted") is not True


def test_market_decision_score_smart_recovery_gating():
    # Caso 1: Sem recovery_active, pontuação padrão
    metrics = {
        "val_accuracy": 0.55,
        "raw_prob": 0.80,
        "edge": 0.30,
        "execute": True,
        "deploy_ok": True,
        "indicators": {
            "adx": 0.15,  # Sem tendência
            "vol_ratio": 0.60,
            "hurst": 0.50,
        },
    }
    # composite calculado com base nos pesos de raw_prob, val_accuracy e edge
    # o valor esperado final e de 0.6925
    score_normal = market_decision_score(metrics, recovery_active=False)
    assert abs(score_normal - 0.6925) < 1e-6

    # Caso 2: Em recovery_active, mas com ADX baixo (< 0.18) -> Aplica penalidade -0.08
    # indicadores: adx=0.15 (< 0.18) -> -0.08
    # mesma direção -> -0.12
    score_recovery_flat = market_decision_score(
        metrics,
        recovery_active=True,
        symbol="R_10",
        exec_direction=TradeDirection.CALL,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    # composite_rec = 0.6925 - 0.12 (mesma dir) - 0.08 (flatline ADX) = 0.4925
    assert abs(score_recovery_flat - 0.4925) < 1e-6

    # Caso 3: Em recovery_active, com ADX alto (>= 0.24) e Vol Ratio alto (>= 1.0) -> Bônus +0.05
    # Hurst > 0.58 -> Bônus +0.03
    metrics_trending = {
        "val_accuracy": 0.55,
        "raw_prob": 0.80,
        "edge": 0.30,
        "execute": True,
        "deploy_ok": True,
        "indicators": {
            "adx": 0.28,  # Forte tendência -> +0.05
            "vol_ratio": 1.10,
            "hurst": 0.62,  # Persistente -> +0.03
        },
    }
    # o valor esperado sob condicoes normais e de 0.6925
    score_recovery_trending = market_decision_score(
        metrics_trending,
        recovery_active=True,
        symbol="R_50",  # Cluster core -> +0.03
        exec_direction=TradeDirection.PUT,
        last_loss_symbol="R_10",  # Diversificação -> +0.04
        last_loss_direction="CALL",  # Direção dif -> +0.03
    )
    # composite_rec = 0.6925 + 0.03 (core) + 0.04 (symbol diff) + 0.03 (dir diff) + 0.05 (ADX/Vol) + 0.03 (Hurst)
    # composite_rec = 0.6925 + 0.18 = 0.8725
    assert abs(score_recovery_trending - 0.8725) < 1e-6

    # Caso 4: Hurst < 0.45 -> Penalidade -0.04
    metrics_erratic = {
        "val_accuracy": 0.55,
        "raw_prob": 0.80,
        "edge": 0.30,
        "execute": True,
        "deploy_ok": True,
        "indicators": {
            "adx": 0.20,
            "vol_ratio": 0.90,
            "hurst": 0.42,  # Ruído errático -> -0.04
        },
    }
    score_recovery_erratic = market_decision_score(
        metrics_erratic,
        recovery_active=True,
        symbol="R_10",
        exec_direction=TradeDirection.PUT,
        last_loss_symbol="R_10",
        last_loss_direction="CALL",  # Direção dif -> +0.03
    )
    # composite_rec = 0.6925 + 0.03 (dir diff) - 0.04 (Hurst) = 0.6825
    assert abs(score_recovery_erratic - 0.6825) < 1e-6
