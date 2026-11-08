"""Compressao da fracao Kelly base fora de recovery."""

from typing import Any

from src.domain.risk.kelly_runtime_config import kelly_runtime_from_config, load_kelly_runtime_from_settings


def _runtime(kelly_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve ou aplica  runtime."""
    if isinstance(kelly_config, dict) and "fraction_base_retention" in kelly_config:
        try:
            return kelly_runtime_from_config({"kelly": kelly_config})
        except ValueError:
            pass
    return load_kelly_runtime_from_settings()


def _compress_kelly_base_fraction(base_fraction: float, runtime: dict[str, Any]) -> float:
    """Resolve ou aplica  compress kelly base fraction."""
    if base_fraction <= 0.0:
        return 0.0
    if abs(base_fraction - float(runtime["fraction_reference"])) < 1e-12:
        return float(runtime["fraction_compressed"])
    return base_fraction * float(runtime["fraction_base_retention"])


def resolve_effective_kelly_fraction(
    kelly_config: dict[str, Any],
    *,
    recovery_active: bool = False,
) -> float:
    """Resolve ou aplica resolve effective kelly fraction."""
    runtime = _runtime(kelly_config)
    base_fraction = float(kelly_config["fraction"]) if "fraction" in kelly_config else float(runtime["fraction"])
    if recovery_active:
        return base_fraction
    explicit_scale = kelly_config.get("kelly_fraction_scale")
    if explicit_scale is not None:
        return base_fraction * max(0.0, float(explicit_scale))
    return _compress_kelly_base_fraction(base_fraction, runtime)
