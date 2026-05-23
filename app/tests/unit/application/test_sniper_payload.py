import numpy as np

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.sniper_payload import (
    _mtf_letter,
    acceleration_token,
    build_mtf_metrics_matrix,
    build_sniper_tokens,
    coerce_sniper_tokens,
    entropy_token,
    format_sniper_prompt_line,
    hurst_token,
    sniper_mtf_bits_from_alignment_sentence,
    velocity_token,
    zscore_token,
)


def test_hurst_token_variants():
    ic = IndicatorConfig(hurst_window=30)
    persist = list(np.linspace(100.0, 110.0, 50))
    assert hurst_token(persist, ic) == "persist"
    anti = [100.0 + (i % 2) * 2.0 for i in range(50)]
    assert hurst_token(anti, ic) == "anti"


def test_zscore_token_variants():
    ic = IndicatorConfig(zscore_window=20)
    flat = [100.0] * 30
    assert zscore_token(flat, ic) == "neutral"
    spike = flat[:-1] + [110.0]
    assert zscore_token(spike, ic) == "high"


def test_entropy_token_variants():
    ic = IndicatorConfig(entropy_window=30, entropy_bins=10)
    clean = [100.0 * (1.001**i) for i in range(50)]
    assert entropy_token(clean, ic) == "low"
    np.random.seed(42)
    noise = (100.0 + np.random.randn(50)).tolist()
    assert entropy_token(noise, ic) == "extreme"


def test_acceleration_token_flat_and_down(monkeypatch):
    ic = IndicatorConfig(acceleration_window=5, velocity_window=5)
    flat = [100.0] * 30
    assert acceleration_token(flat, ic) == "flat"
    monkeypatch.setattr(
        "src.application.services.llm.sniper_payload.ti._price_derivatives",
        lambda _c, _w: (0.1, -0.01),
    )
    assert acceleration_token(flat, ic) == "down"


def test_build_mtf_metrics_matrix():
    ic = IndicatorConfig()
    closes = list(np.linspace(100.0, 110.0, 40))
    line = build_mtf_metrics_matrix([("M15", closes), ("M1", closes)], ic, None)
    assert line.startswith("MTF_MATRIX:")
    assert "M15[" in line


def test_velocity_token_variants():
    ic = IndicatorConfig(velocity_window=5)
    up = [100, 101, 102, 103, 104, 105, 106, 107]
    assert velocity_token(up, ic) == "pos"
    down = [107, 106, 105, 104, 103, 102, 101, 100]
    assert velocity_token(down, ic) == "neg"


def test_coerce_sniper_tokens():
    empty = {"hurst": "na", "zscore": "na", "entropy": "na", "velocity": "na", "acceleration": "na"}
    assert coerce_sniper_tokens(None) == empty
    assert coerce_sniper_tokens({"hurst": "persist", "zscore": "high"}) == {
        "hurst": "persist",
        "zscore": "high",
        "entropy": "na",
        "velocity": "na",
        "acceleration": "na",
    }


def test_mtf_letter_regime():
    assert _mtf_letter("Momentum Alpha (Bull)") == "P"
    assert _mtf_letter("Mean Reversion Alpha") == "M"
    assert _mtf_letter("random noise market") == "N"


def test_build_and_format_sniper_line():
    ic = IndicatorConfig()
    closes = list(np.linspace(100.0, 110.0, 120))
    tok = build_sniper_tokens(closes, ic, None)
    assert set(tok.keys()) == {"hurst", "zscore", "entropy", "velocity", "acceleration"}
    line = format_sniper_prompt_line(
        "OTC_FCHI",
        "trend_persistente",
        "random_walk",
        "mean_reverting",
        "noise",
        coerce_sniper_tokens(tok),
        mtf_alignment_line="M30: momentum | M15: random | M5: mean | M1: noise",
    )
    assert "h=" in line
    assert "MTF=P/N/M/N" in line
    assert "SYM=OTC_FCHI" in line


def test_sniper_mtf_bits_from_alignment_sentence():
    assert sniper_mtf_bits_from_alignment_sentence("M30: persistence | M15: mean | M5: noise") == "P/M/N"
