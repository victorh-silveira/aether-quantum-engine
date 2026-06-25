"""Cooldown entre operacoes (ticks e tempo real)."""

from typing import Any


class RiskCooldownMixin:
    """Metodos de cooldown desativados para RiskManager."""

    config: dict[str, Any]
    risk_params: dict[str, Any]
    base_cooldown: int
    current_cooldown_ticks: int
    _last_entry_conviction: float
    _candle_interval_seconds: int
    _cooldown_until_mono: float
    last_result_tick: int

    def register_entry_conviction(self, conviction: float) -> None:
        """Registra conviccao (no-op)."""
        _ = conviction

    def effective_cooldown_ticks(self) -> int:
        """Sempre 0 (desativado)."""
        return 0

    def _uses_seconds_cooldown(self) -> bool:
        """Sempre False (desativado)."""
        return False

    def _cooldown_span_seconds(self) -> float:
        """Sempre 0.0 (desativado)."""
        return 0.0

    def _arm_cooldown_timer(self) -> None:
        """No-op."""
        pass

    def cooldown_remaining_seconds(self) -> float:
        """Sempre 0.0 (desativado)."""
        return 0.0

    def is_on_cooldown(self, current_tick: int) -> bool:
        """Sempre False (desativado)."""
        _ = current_tick
        return False
