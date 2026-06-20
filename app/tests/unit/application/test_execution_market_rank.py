from src.application.services.execution_market_rank import (
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def test_resolve_market_direction_uses_entry_direction():
    entry = {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.80}}
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_market_decision_score_uses_raw_and_val_accuracy():
    metrics = {
        "trade_score": 0.80,
        "raw_prob": 0.80,
        "val_accuracy": 0.55,
        "execute": True,
        "deploy_ok": True,
    }
    score = market_decision_score(
        metrics,
        exec_direction=TradeDirection.CALL,
        recovery_active=True,
        symbol="R_50",
        last_loss_symbol="R_10",
        last_loss_direction="CALL",
    )
    assert score > 0.5


def test_mandatory_pool_eligible_rejects_data_gate():
    entry = {"direction": TradeDirection.CALL, "metrics": {"gate_reason": "data"}}
    assert mandatory_pool_eligible(entry) is False
    assert mandatory_pool_eligible({"direction": TradeDirection.PUT, "metrics": {"raw_prob": 0.20}}) is True


def test_build_market_execution_candidate():
    entry = {"direction": TradeDirection.PUT, "metrics": {"raw_prob": 0.20, "execute": True}}
    built = build_market_execution_candidate("R_25", entry)
    assert built is not None
    assert built[1] == TradeDirection.PUT


def test_resolve_market_direction_low_accuracy_inverts():
    # Acurácia de validação < 50% inverte CALL para PUT
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.45, "raw_prob": 0.80},
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT
    assert entry["metrics"]["direction_inverted"] is True

    # Acurácia de validação < 50% inverte PUT para CALL
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"val_accuracy": 0.40, "raw_prob": 0.20},
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL
    assert entry["metrics"]["direction_inverted"] is True


def test_resolve_market_direction_follows_trend_on_gate_blocked():
    # Com acurácia >= 50% e execute=False, se houver trend_direction, deve seguir a tendência
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.80,
            "execute": False,
            "trend_direction": "PUT",
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT


def test_build_market_execution_candidate_preserves_inverted_flag():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.45, "raw_prob": 0.80, "execute": True},
    }
    built = build_market_execution_candidate("R_25", entry)
    assert built is not None
    assert built[1] == TradeDirection.PUT
    assert built[2]["direction_inverted"] is True


def test_resolve_market_direction_invalid_trend_direction():
    # Com acurácia >= 50% e execute=False, se houver trend_direction inválida, deve ignorar e retornar dl_dir
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.52,
            "raw_prob": 0.80,
            "execute": False,
            "trend_direction": "INVALID_TREND_NAME",
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.CALL


def test_resolve_market_direction_recovery_trend_alignment():
    # Em recuperação com perdas consecutivas, deve usar a tendência SMA se disponível
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.55,
            "raw_prob": 0.80,
            "execute": True,
            "trend_direction": "PUT",
        },
    }
    assert resolve_market_direction(entry, recovery_active=True, consecutive_losses=1) == TradeDirection.PUT
    assert entry["metrics"]["direction_inverted"] is True


def test_resolve_market_direction_low_accuracy_uses_trend_if_available():
    # Baixa acurácia (< 0.50) com tendência disponível deve usar a tendência diretamente
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.45,
            "raw_prob": 0.80,
            "execute": True,
            "trend_direction": "PUT",
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT
    assert entry["metrics"]["direction_inverted"] is True


def test_build_market_execution_candidate_preserves_dl_and_exec_directions():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"val_accuracy": 0.45, "raw_prob": 0.80, "execute": True},
    }
    built = build_market_execution_candidate("R_25", entry)
    assert built is not None
    assert built[2]["dl_direction"] == "CALL"
    assert built[2]["exec_direction"] == "PUT"
    assert built[2]["direction_inverted"] is True


def test_resolve_market_direction_invalid_trend_low_accuracy():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.45,
            "raw_prob": 0.80,
            "trend_direction": "INVALID_VAL",
        },
    }
    # Deve capturar a exceção e fazer a inversão padrão (CALL -> PUT)
    assert resolve_market_direction(entry) == TradeDirection.PUT
    assert entry["metrics"]["direction_inverted"] is True


def test_resolve_market_direction_invalid_trend_recovery():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "val_accuracy": 0.55,
            "raw_prob": 0.80,
            "trend_direction": "INVALID_VAL",
        },
    }
    # Deve capturar a exceção e retornar o dl_dir (CALL)
    assert resolve_market_direction(entry, recovery_active=True, consecutive_losses=1) == TradeDirection.CALL
