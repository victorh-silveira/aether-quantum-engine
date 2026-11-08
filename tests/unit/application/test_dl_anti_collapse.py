"""Testes de garantia contra colapso de predicao 50%/50% na TCN e calibracao."""

import numpy as np
import pytest
import torch

from src.application.services.deep_learning.dl_calibration import (
    CalibratorState,
    apply_calibrator_stable,
)
from src.application.services.deep_learning.dl_calibration_fit import (
    fit_calibrator,
    maybe_identity_on_oos_collapse,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_sharpness import mean_sharpness
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import FeatureNormStats, predict_next_direction
from src.domain.models.trade import TradeDirection


def test_tcn_forward_sharp_outputs_on_directional_signal():
    """TCN causal com attention pooling reage assimetricamente a padroes de alta e baixa."""
    model = TemporalDirectionClassifier(input_dim=FEATURE_DIM)
    model.eval()

    up_seq = torch.full((1, 32, FEATURE_DIM), 0.8, dtype=torch.float32)
    down_seq = torch.full((1, 32, FEATURE_DIM), -0.8, dtype=torch.float32)

    with torch.inference_mode():
        p_up = model(up_seq).item()
        p_down = model(down_seq).item()

    assert p_up != pytest.approx(0.5, abs=0.001) or p_down != pytest.approx(0.5, abs=0.001)
    assert 0.0 <= p_up <= 1.0
    assert 0.0 <= p_down <= 1.0


def test_calibration_fit_preserves_sharpness_floor():
    """O ajuste do calibrador nunca seleciona metodo que degrade sharpness abaixo do piso."""
    raw_probs = [0.20, 0.25, 0.75, 0.80, 0.22, 0.78] * 10
    labels = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0] * 10
    cfg = {"min_calibration_sharpness": 0.06, "min_oos_sharpness": 0.06}

    calibrator = fit_calibrator(raw_probs, labels, calibration_cfg=cfg)
    calibrated = [apply_calibrator_stable(p, calibrator) for p in raw_probs]
    sharp = mean_sharpness(calibrated)

    assert sharp >= 0.06
    assert abs(sharp - 0.0) > 1e-4


def test_maybe_identity_on_oos_collapse_triggers_when_calibrator_flattens():
    """Quando calibrador tenta achatar para 0.50, reverte para identity mantendo nitidez."""
    val_probs = [0.42, 0.58] * 10
    flattening_calibrator = CalibratorState(
        method="temperature",
        temperature=1.5,
        platt_a=1.0,
        platt_b=0.0,
    )
    chosen, sharp = maybe_identity_on_oos_collapse(
        flattening_calibrator,
        val_probs=val_probs,
        min_oos_sharpness=0.06,
    )
    assert chosen.method == "identity"
    assert sharp >= 0.06


def test_predict_next_direction_never_stalls_at_fifty_on_trend():
    """Garante que a predicao do modelo sob serie de precos com drift convicto nao retorna 0.50."""
    model = TemporalDirectionClassifier(input_dim=FEATURE_DIM)
    model.eval()

    prices_up = np.linspace(100.0, 150.0, 64, dtype=np.float64)
    norm_stats = FeatureNormStats(
        mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        std=np.ones(FEATURE_DIM, dtype=np.float32),
    )

    side, prob, raw_prob = predict_next_direction(
        model,
        prices_up,
        lookback=32,
        norm_stats=norm_stats,
    )
    assert side in (TradeDirection.CALL, TradeDirection.PUT)
    assert abs(raw_prob - 0.5) > 1e-5
    assert abs(prob - 0.5) > 1e-5


def test_predict_next_direction_recovers_when_calibrator_flat_but_model_confident():
    """Garante que calibrador plano e modelo convicto reativa dispersao direcional."""
    from unittest.mock import patch

    import pytest

    model = TemporalDirectionClassifier(input_dim=FEATURE_DIM)
    model.eval()

    flat_calibrator = CalibratorState(
        method="temperature",
        temperature=100.0,
        platt_a=0.0,
        platt_b=0.0,
    )
    prices_up = np.linspace(100.0, 150.0, 64, dtype=np.float64)
    norm_stats = FeatureNormStats(
        mean=np.zeros(FEATURE_DIM, dtype=np.float32),
        std=np.ones(FEATURE_DIM, dtype=np.float32),
    )
    with patch(
        "src.application.services.deep_learning.model._model_raw_prob_and_aux",
        return_value=(np.array([0.65], dtype=np.float32), np.array([0.0], dtype=np.float32)),
    ):
        side, prob, raw_prob = predict_next_direction(
            model,
            prices_up,
            lookback=32,
            norm_stats=norm_stats,
            calibrator=flat_calibrator,
        )
    assert side == TradeDirection.CALL
    assert prob == pytest.approx(0.65)
    assert raw_prob == pytest.approx(0.65)
