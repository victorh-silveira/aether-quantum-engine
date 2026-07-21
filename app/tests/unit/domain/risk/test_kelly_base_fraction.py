import pytest

from src.domain.risk.kelly_base_fraction import resolve_effective_kelly_fraction
from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings


def test_resolve_effective_kelly_fraction_attenuates_normal_regime():
    runtime = load_kelly_runtime_from_settings()
    reference = float(runtime["fraction_reference"])
    compressed = float(runtime["fraction_compressed"])
    retention = float(runtime["fraction_base_retention"])
    cfg = {"fraction": reference}
    assert resolve_effective_kelly_fraction(cfg, recovery_active=False) == pytest.approx(compressed)
    assert resolve_effective_kelly_fraction(cfg, recovery_active=True) == pytest.approx(reference)
    assert pytest.approx(0.40) == retention


def test_resolve_effective_kelly_fraction_sixty_percent_reduction_generic():
    cfg = {"fraction": 0.005}
    assert resolve_effective_kelly_fraction(cfg, recovery_active=False) == pytest.approx(0.002)
    assert resolve_effective_kelly_fraction({"fraction": 0.0}, recovery_active=False) == pytest.approx(0.0)


def test_resolve_effective_kelly_fraction_honors_explicit_scale():
    cfg = {"fraction": 0.0035, "kelly_fraction_scale": 0.80}
    assert resolve_effective_kelly_fraction(cfg, recovery_active=False) == pytest.approx(0.0028)
