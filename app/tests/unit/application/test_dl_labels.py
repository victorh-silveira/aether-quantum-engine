import numpy as np

from src.application.services.deep_learning.dl_horizon import (
    resolve_implied_vol_bars,
    resolve_label_horizon_bars,
    resolve_label_ma_window,
    resolve_label_mode,
    resolve_label_smooth_bars,
)
from src.application.services.deep_learning.dl_labels import (
    LABEL_MODE_MA_TREND,
    LABEL_MODE_SPOT,
    LabelSpec,
    binary_label_at_index,
    sequence_labels,
)


def test_label_horizon_one_bar_for_60s_contract():
    horizon = resolve_label_horizon_bars(60, {"duration": 60, "duration_unit": "s"}, {})
    assert horizon == 1


def test_label_horizon_one_bar_for_900s_contract_m15():
    horizon = resolve_label_horizon_bars(900, {"duration": 900, "duration_unit": "s"}, {})
    assert horizon == 1


def test_binary_label_rise_spot():
    prices = np.array([100.0, 101.0, 99.0], dtype=np.float64)
    assert binary_label_at_index(prices, 0, 1, smooth_bars=1, label_mode=LABEL_MODE_SPOT) is True
    assert binary_label_at_index(prices, 1, 1, smooth_bars=1, label_mode=LABEL_MODE_SPOT) is False


def test_binary_label_ma_trend():
    prices = np.array([100.0, 100.0, 100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64)
    assert binary_label_at_index(prices, 2, 1, smooth_bars=3, label_mode=LABEL_MODE_MA_TREND, ma_window=3) is True
    prices_down = np.array([104.0, 103.0, 102.0, 101.0, 100.0, 99.0, 98.0], dtype=np.float64)
    assert binary_label_at_index(prices_down, 2, 1, smooth_bars=2, label_mode=LABEL_MODE_MA_TREND, ma_window=3) is False


def test_resolve_label_helpers():
    assert resolve_label_smooth_bars({}) == 1
    assert resolve_label_smooth_bars({"label_smooth_bars": 5}) == 5
    assert resolve_label_ma_window({"label_ma_window": 8}) == 8
    assert resolve_label_mode({}) == "quantum_multi_barrier"
    assert resolve_label_mode({"label_mode": "quantum"}) == "quantum_multi_barrier"
    assert resolve_label_mode({"label_mode": "quantum_multi_barrier"}) == "quantum_multi_barrier"
    assert resolve_label_mode({"label_mode": "multi_barrier"}) == "quantum_multi_barrier"
    assert resolve_label_mode({"label_mode": "qmb"}) == "quantum_multi_barrier"
    assert resolve_label_mode({"label_mode": "triple_barrier"}) == "triple_barrier"
    assert resolve_label_mode({"label_mode": "triple"}) == "triple_barrier"
    assert resolve_label_mode({"label_mode": "spot_forward"}) == "spot_forward"
    assert resolve_label_mode({"label_mode": "supertrend"}) == "supertrend_atr"
    assert resolve_label_mode({"label_mode": "ma_trend"}) == "ma_trend"
    assert resolve_implied_vol_bars({"implied_vol_bars": 60}) == 60


def test_sequence_labels_shape():
    prices = np.linspace(100.0, 120.0, 80)
    targets, masks = sequence_labels(prices, lookback=48, horizon_bars=1, smooth_bars=1)
    assert len(targets) == len(masks) == 80 - 48 - 1
    assert masks.sum() == len(masks)


def test_sequence_labels_shape_with_smooth():
    prices = np.linspace(100.0, 120.0, 80)
    targets, masks = sequence_labels(prices, lookback=48, horizon_bars=1, smooth_bars=5)
    assert len(targets) == len(masks) == 80 - 48 - 5


def test_label_spec_embargo_bars():
    spec = LabelSpec(horizon_bars=1, smooth_bars=5)
    assert spec.embargo_bars == 5
    assert LabelSpec.from_dl_config({"label_horizon_bars": 2, "label_smooth_bars": 3}).embargo_bars == 4


def test_binary_label_supertrend_atr():
    from src.application.services.deep_learning.dl_labels import LABEL_MODE_SUPERTREND_ATR, _supertrend_direction

    prices = np.linspace(100.0, 150.0, 50)
    assert _supertrend_direction(prices, 5) == 1
    assert _supertrend_direction(prices, 40) == 1
    assert binary_label_at_index(prices, 20, 5, label_mode=LABEL_MODE_SUPERTREND_ATR) is True

    prices_down = np.linspace(150.0, 100.0, 50)
    assert _supertrend_direction(prices_down, 40) == -1
    assert binary_label_at_index(prices_down, 20, 5, label_mode=LABEL_MODE_SUPERTREND_ATR) is False
    prices_flat = np.full(50, 100.0)
    assert _supertrend_direction(prices_flat, 40) == 1
    prices_flat_step = np.array([100.0] * 30 + [100.1, 100.0])
    assert _supertrend_direction(prices_flat_step, 31) == -1


def test_binary_label_triple_barrier():
    from src.application.services.deep_learning.dl_labels import (
        LABEL_MODE_TRIPLE_BARRIER,
        _triple_barrier_direction,
    )

    prices_up = np.linspace(100.0, 150.0, 50)
    assert _triple_barrier_direction(prices_up, 20, 5) is True
    assert binary_label_at_index(prices_up, 20, 5, label_mode=LABEL_MODE_TRIPLE_BARRIER) is True

    prices_down = np.linspace(150.0, 100.0, 50)
    assert _triple_barrier_direction(prices_down, 20, 5) is False
    assert binary_label_at_index(prices_down, 20, 5, label_mode=LABEL_MODE_TRIPLE_BARRIER) is False

    # Primeiro toque inferior rápido
    prices_touch_lower = np.array([100.0] * 20 + [100.0, 95.0, 105.0, 105.0])
    assert _triple_barrier_direction(prices_touch_lower, 20, 3) is False

    # Primeiro toque superior rápido
    prices_touch_upper = np.array([100.0] * 20 + [100.0, 105.0, 95.0, 95.0])
    assert _triple_barrier_direction(prices_touch_upper, 20, 3) is True

    # Sem tocar barreiras, fechando no mesmo nivel
    prices_flat = np.full(50, 100.0)
    assert _triple_barrier_direction(prices_flat, 20, 2) is True

    # Teste de vol_seg <= 1 (linha 94)
    assert _triple_barrier_direction(prices_flat[:1], 0, 1) is True


def test_forward_mean_out_of_bounds_and_empty_sequence():
    # Linha 55: forward_end > len(prices)
    prices = np.array([100.0, 101.0, 102.0])
    assert binary_label_at_index(prices, index=2, horizon_bars=5) is False

    # Linha 156: sequence_labels com array curto
    short_prices = np.array([100.0, 101.0])
    t, m = sequence_labels(short_prices, lookback=10, horizon_bars=1)
    assert len(t) == 0 and len(m) == 0


def test_triple_barrier_candlestick_patterns_and_noise_invariance():
    from src.application.services.deep_learning.dl_labels import (
        LABEL_MODE_TRIPLE_BARRIER,
        _triple_barrier_direction,
    )

    # Forte impulso de alta (Marubozu de alta): atinge barreira superior imediatamente
    prices_marubozu_bull = np.array([100.0] * 20 + [100.0, 108.0, 115.0])
    assert _triple_barrier_direction(prices_marubozu_bull, 20, 2) is True
    assert binary_label_at_index(prices_marubozu_bull, 20, 2, label_mode=LABEL_MODE_TRIPLE_BARRIER) is True

    # Forte rejeição de baixa (Marubozu de baixa): atinge barreira inferior imediatamente
    prices_marubozu_bear = np.array([100.0] * 20 + [100.0, 92.0, 85.0])
    assert _triple_barrier_direction(prices_marubozu_bear, 20, 2) is False
    assert binary_label_at_index(prices_marubozu_bear, 20, 2, label_mode=LABEL_MODE_TRIPLE_BARRIER) is False

    # Candle de indecisão (Doji) com micro ruído: avalia barreira temporal no horizonte
    prices_doji = np.array([100.0] * 20 + [100.0, 100.01, 100.02])
    assert _triple_barrier_direction(prices_doji, 20, 2) is True


def test_quantum_multi_barrier_direction():
    from src.application.services.deep_learning.dl_labels import (
        LABEL_MODE_QUANTUM_MULTI_BARRIER,
        _quantum_multi_barrier_direction,
    )

    # Caso 1: Impulso de alta rompendo barreira superior assimétrica
    prices_up = np.array([100.0] * 20 + [100.0, 105.0, 110.0])
    assert _quantum_multi_barrier_direction(prices_up, 20, 2) is True
    assert binary_label_at_index(prices_up, 20, 2, label_mode=LABEL_MODE_QUANTUM_MULTI_BARRIER) is True

    # Caso 2: Impulso de baixa rompendo barreira inferior assimétrica
    prices_down = np.array([100.0] * 20 + [100.0, 95.0, 90.0])
    assert _quantum_multi_barrier_direction(prices_down, 20, 2) is False
    assert binary_label_at_index(prices_down, 20, 2, label_mode=LABEL_MODE_QUANTUM_MULTI_BARRIER) is False

    # Caso 3: Expiração na barreira vertical com deslocamento positivo expressivo sem tocar a barreira dinâmica
    prices_exp_call = np.array([100.0] * 20 + [100.0, 100.005])
    assert _quantum_multi_barrier_direction(prices_exp_call, 20, 1, min_viable_delta=0.00001, barrier_mult=50.0) is True

    # Caso 4: Expiração na barreira vertical com deslocamento negativo expressivo sem tocar a barreira dinâmica
    prices_exp_put = np.array([100.0] * 20 + [100.0, 99.995])
    assert _quantum_multi_barrier_direction(prices_exp_put, 20, 1, min_viable_delta=0.00001, barrier_mult=50.0) is False

    # Caso 5: Consolidação estagnada (delta < threshold) desempatada pela tendência prévia (alta)
    prices_flat_uptrend = np.array(list(np.linspace(95.0, 100.0, 20)) + [100.0, 100.0, 100.0])
    assert _quantum_multi_barrier_direction(prices_flat_uptrend, 20, 2) is True

    # Caso 6: Consolidação estagnada (delta < threshold) desempatada pela tendência prévia (baixa)
    prices_flat_downtrend = np.array(list(np.linspace(105.0, 100.0, 20)) + [100.0, 100.0, 100.0])
    assert _quantum_multi_barrier_direction(prices_flat_downtrend, 20, 2) is False

    # Caso 7: Array de volatilidade curto
    prices_short = np.array([100.0, 100.1])
    assert _quantum_multi_barrier_direction(prices_short, 0, 1) is True
