"""Testes unitarios para inferencia de probabilidade e movimento no deep learning."""

import numpy as np
import torch
from torch import nn

from src.application.services.deep_learning.model import (
    _model_raw_prob_and_aux,
    predict_direction_and_movement,
    predict_next_direction,
)
from src.domain.models.trade import TradeDirection


class DummyDualHeadModel(nn.Module):
    """Modelo dummy com cabeca de classificacao e regressao."""

    def __init__(self, prob_val: float = 0.65, delta_val: float = 0.0025):
        super().__init__()
        self.prob = prob_val
        self.delta = delta_val
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, *args, return_aux: bool = False, **kwargs):
        _ = (args, kwargs)
        batch = x.shape[0]
        p = torch.full((batch, 1), self.prob, dtype=torch.float32, device=x.device)
        aux = torch.full((batch, 1), self.delta, dtype=torch.float32, device=x.device)
        if return_aux:
            return p, aux
        return p


class DummySingleHeadModel(nn.Module):
    """Modelo dummy legado de cabeca unica."""

    def __init__(self, prob_val: float = 0.40):
        super().__init__()
        self.prob = prob_val
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        return torch.full((batch, 1), self.prob, dtype=torch.float32, device=x.device)


def test_model_raw_prob_and_aux_dual():
    model = DummyDualHeadModel(0.70, 0.005)
    batch = np.zeros((2, 10, 14), dtype=np.float32)
    probs, auxes = _model_raw_prob_and_aux(model, batch)
    assert len(probs) == 2
    assert len(auxes) == 2
    assert abs(probs[0] - 0.70) < 1e-5
    assert abs(auxes[0] - 0.005) < 1e-5
    assert hasattr(model, "_last_predicted_delta")
    assert abs(model._last_predicted_delta - 0.005) < 1e-5


def test_model_raw_prob_and_aux_single():
    model = DummySingleHeadModel(0.35)
    batch = np.zeros((1, 5, 14), dtype=np.float32)
    probs, auxes = _model_raw_prob_and_aux(model, batch)
    assert len(probs) == 1
    assert abs(probs[0] - 0.35) < 1e-5
    assert auxes[0] == 0.0
    assert model._last_predicted_delta == 0.0


def test_predict_next_direction_movement_flag():
    model = DummyDualHeadModel(0.60, 0.004)
    prices = np.linspace(100.0, 110.0, 40)
    lookback = 10

    side, prob, raw = predict_next_direction(model, prices, lookback, return_movement=False)
    assert side == TradeDirection.CALL
    assert abs(prob - 0.60) < 1e-4

    side2, prob2, raw2, delta2 = predict_next_direction(model, prices, lookback, return_movement=True)
    assert side2 == TradeDirection.CALL
    assert abs(prob2 - 0.60) < 1e-4
    assert abs(delta2 - 0.004) < 1e-4


def test_predict_direction_and_movement_helper():
    model = DummyDualHeadModel(0.30, -0.006)
    prices = np.linspace(110.0, 100.0, 40)
    lookback = 10

    side, prob, raw, delta = predict_direction_and_movement(model, prices, lookback)
    assert side == TradeDirection.PUT
    assert abs(prob - 0.30) < 1e-4
    assert abs(delta - (-0.006)) < 1e-4


def test_predict_next_direction_insufficient_data():
    model = DummyDualHeadModel()
    prices = np.array([100.0, 101.0])
    lookback = 10

    res = predict_next_direction(model, prices, lookback, return_movement=False)
    assert res == (None, 0.5, 0.5)

    res_mv = predict_next_direction(model, prices, lookback, return_movement=True)
    assert res_mv == (None, 0.5, 0.5, 0.0)
