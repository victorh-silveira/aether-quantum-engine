import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.application.services.deep_learning.decision_bridge import collect_deep_learning_decisions
from src.application.services.deep_learning.model import (
    MarketDirectionClassifier,
    calculate_rsi,
    extract_features,
    predict_next_direction,
    train_model_online,
)
from src.domain.models.trade import TradeDirection


def test_model_initialization():
    model = MarketDirectionClassifier(input_dim=4)
    x = torch.randn(2, 4)
    out = model(x)
    assert out.shape == (2, 1)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_calculate_rsi():
    prices = np.linspace(10, 20, 30)  # Preço subindo uniformemente
    rsi = calculate_rsi(prices, period=14)
    assert len(rsi) == 30
    assert rsi[-1] > 70.0  # RSI deve ser alto para tendência de alta constante

    # Cobertura: Linha 32 em model.py (preços menores que período)
    rsi_short = calculate_rsi(np.array([1.0, 2.0]), period=14)
    assert len(rsi_short) == 2
    assert np.all(rsi_short == 50.0)


def test_extract_features():
    prices = np.sin(np.linspace(0, 10, 50)) + 10.0
    features, targets = extract_features(prices, lookback=20)
    assert features.shape[0] == targets.shape[0]
    if features.shape[0] > 0:
        assert features.shape[1] == 4

    # Cobertura: Linha 59 em model.py (preços insuficientes para features)
    f, t = extract_features(np.array([10.0, 11.0]), lookback=20)
    assert len(f) == 0
    assert len(t) == 0


def test_train_and_predict():
    prices = np.sin(np.linspace(0, 10, 60)) + 10.0
    model = MarketDirectionClassifier(input_dim=4)
    loss = train_model_online(model, prices, lookback=20, epochs=2, lr=0.01)
    assert isinstance(loss, float)

    # Cobertura: Linha 91 em model.py (preços insuficientes para treino)
    loss_short = train_model_online(model, np.sin(np.linspace(0, 10, 25)) + 10.0, lookback=20, epochs=2, lr=0.01)
    assert loss_short == 0.0

    # Cobertura: Garantir execução de ambas as ramificações de probabilidade (CALL e PUT)
    with patch.object(model, "forward", return_value=torch.tensor([[0.8]])):
        direction, prob = predict_next_direction(model, prices, lookback=20)
        assert direction == TradeDirection.CALL
        assert prob == pytest.approx(0.8)

    with patch.object(model, "forward", return_value=torch.tensor([[0.3]])):
        direction, prob = predict_next_direction(model, prices, lookback=20)
        assert direction == TradeDirection.PUT
        assert prob == pytest.approx(0.7)

    # Cobertura: Linha 117 em model.py (preços insuficientes para predição)
    dir_short, prob_short = predict_next_direction(model, np.array([10.0]), lookback=20)
    assert dir_short is None
    assert prob_short == 0.5


class MockStreamHandler:
    def __init__(self, prices):
        self.prices = prices

    def get_numpy_series(self, _symbol, _field):
        return self.prices


class MockOrchestrator:
    def __init__(self, symbols, prices, *, dl_enabled=True):
        self.symbols = symbols
        self.config = {
            "deep_learning": {
                "enabled": dl_enabled,
                "lookback": 15,
                "training_epochs": 1,
                "learning_rate": 0.01,
                "min_conviction_execute": 0.50,
                "model_path": "nonexistent_model.pth",
            }
        }
        self.stream = MockStreamHandler(prices)


@pytest.mark.asyncio
async def test_collect_deep_learning_decisions():
    prices = np.sin(np.linspace(0, 10, 50)) + 10.0
    orch = MockOrchestrator(["1HZ100V", "1HZ75V"], prices)
    decisions = await collect_deep_learning_decisions(orch)
    assert "1HZ100V" in decisions
    assert "1HZ75V" in decisions
    assert "direction" in decisions["1HZ100V"]
    assert "metrics" in decisions["1HZ100V"]
    assert "conviction" in decisions["1HZ100V"]["metrics"]

    # Cobertura: Linhas 20-21 em decision_bridge.py (deep_learning disabled)
    orch_disabled = MockOrchestrator(["1HZ100V"], prices, dl_enabled=False)
    dec_disabled = await collect_deep_learning_decisions(orch_disabled)
    assert dec_disabled == {}

    # Cobertura: Linhas 36-46 em decision_bridge.py (insufficient prices)
    orch_short = MockOrchestrator(["1HZ100V"], np.array([1.0, 2.0]), dl_enabled=True)
    dec_short = await collect_deep_learning_decisions(orch_short)
    assert dec_short["1HZ100V"]["direction"] is None
    assert dec_short["1HZ100V"]["metrics"]["conviction"] == 0.0


@pytest.mark.asyncio
async def test_collect_decisions_exceptions_and_load():
    prices = np.sin(np.linspace(0, 10, 50)) + 10.0
    orch = MockOrchestrator(["1HZ100V"], prices)

    # 1. Testar carregamento de pesos com sucesso
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model = MarketDirectionClassifier(input_dim=4)
        torch.save(model.state_dict(), tmp_path)
        orch.config["deep_learning"]["model_path"] = tmp_path

        # Garante que recria o modelo para passar pela rotina de carregamento
        if hasattr(orch, "_dl_models"):
            orch._dl_models.clear()

        decisions = await collect_deep_learning_decisions(orch)
        assert "1HZ100V" in decisions
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # 2. Testar falha de carregamento de pesos (arquivo existe mas é inválido)
    orch.config["deep_learning"]["model_path"] = __file__
    if hasattr(orch, "_dl_models"):
        orch._dl_models.clear()
    decisions = await collect_deep_learning_decisions(orch)
    assert "1HZ100V" in decisions

    # Cobertura: Linhas 67-68 em decision_bridge.py (train exception)
    with patch(
        "src.application.services.deep_learning.decision_bridge.train_model_online",
        side_effect=ValueError("Train failed"),
    ):
        dec = await collect_deep_learning_decisions(orch)
        assert "1HZ100V" in dec

    # Cobertura: Linhas 86-88 em decision_bridge.py (predict exception)
    with patch(
        "src.application.services.deep_learning.decision_bridge.predict_next_direction",
        side_effect=ValueError("Predict failed"),
    ):
        dec = await collect_deep_learning_decisions(orch)
        assert dec["1HZ100V"]["direction"] is None
        assert dec["1HZ100V"]["metrics"]["conviction"] == 0.0
