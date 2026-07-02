import pytest

from src.domain.risk.kelly_base_fraction import (
    KELLY_FRACTION_BASE_RETENTION,
    KELLY_FRACTION_COMPRESSED,
    KELLY_FRACTION_REFERENCE,
    resolve_effective_kelly_fraction,
)


def test_resolve_effective_kelly_fraction_attenuates_normal_regime():
    cfg = {"fraction": KELLY_FRACTION_REFERENCE}
    assert resolve_effective_kelly_fraction(cfg, recovery_active=False) == pytest.approx(KELLY_FRACTION_COMPRESSED)
    assert resolve_effective_kelly_fraction(cfg, recovery_active=True) == pytest.approx(KELLY_FRACTION_REFERENCE)
    assert pytest.approx(0.40) == KELLY_FRACTION_BASE_RETENTION


def test_resolve_effective_kelly_fraction_sixty_percent_reduction_generic():
    cfg = {"fraction": 0.005}
    assert resolve_effective_kelly_fraction(cfg, recovery_active=False) == pytest.approx(0.002)
    assert resolve_effective_kelly_fraction({"fraction": 0.0}, recovery_active=False) == pytest.approx(0.0)


def test_resolve_effective_kelly_fraction_honors_explicit_scale():
    cfg = {"fraction": 0.0035, "kelly_fraction_scale": 0.80}
    assert resolve_effective_kelly_fraction(cfg, recovery_active=False) == pytest.approx(0.0028)
