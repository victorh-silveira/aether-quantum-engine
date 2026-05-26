"""Cooldown entre operacoes (ticks e tempo real)."""

import time
from typing import Any

from src.domain.risk.entry_cooldown import resolve_entry_cooldown_seconds, resolve_entry_cooldown_ticks


class RiskCooldownMixin:
    """Metodos de cooldown para RiskManager."""

    config: dict[str, Any]
    risk_params: dict[str, Any]
    base_cooldown: int
    current_cooldown_ticks: int
    _last_entry_conviction: float
    _candle_interval_seconds: int
    _cooldown_until_mono: float
    last_result_tick: int

    def register_entry_conviction(self, conviction: float) -> None:
        """Registra conviccao da ultima entrada para cooldown dinamico."""
        self._last_entry_conviction = max(0.0, float(conviction))

    def effective_cooldown_ticks(self) -> int:
        """Cooldown efetivo apos cluster (dinamico por conviccao da ultima entrada)."""
        active = int(self.current_cooldown_ticks)
        if active <= 0:
            return 0
        target = resolve_entry_cooldown_ticks(self.config, self._last_entry_conviction)
        if target <= 0:
            return active
        return min(active, target)

    def _uses_seconds_cooldown(self) -> bool:
        """True quando entry_cooldown_seconds esta configurado."""
        return resolve_entry_cooldown_seconds(self.config, self._last_entry_conviction) is not None

    def _cooldown_span_seconds(self) -> float:
        """Duracao em segundos do cooldown ativo apos liquidacao."""
        secs = resolve_entry_cooldown_seconds(self.config, self._last_entry_conviction)
        if secs is not None:
            return float(secs)
        ticks = self.effective_cooldown_ticks()
        if ticks <= 0:
            return 0.0
        tick_secs = self.risk_params.get("entry_cooldown_tick_seconds")
        if tick_secs is not None:
            return ticks * max(1.0, float(tick_secs))
        return ticks * float(self._candle_interval_seconds)

    def _arm_cooldown_timer(self) -> None:
        """Inicia ou limpa o timer monotonic de cooldown."""
        span = self._cooldown_span_seconds()
        if span <= 0:
            self._cooldown_until_mono = 0.0
            return
        self._cooldown_until_mono = time.monotonic() + span

    def cooldown_remaining_seconds(self) -> float:
        """Segundos restantes de cooldown por tempo real."""
        if self._cooldown_until_mono <= 0:
            return 0.0
        return max(0.0, self._cooldown_until_mono - time.monotonic())

    def is_on_cooldown(self, current_tick: int) -> bool:
        """Cooldown entre operacoes (tempo real + fallback por ticks)."""
        if self._uses_seconds_cooldown():
            return self.cooldown_remaining_seconds() > 0
        need = self.effective_cooldown_ticks()
        if need <= 0:
            return False
        if self._cooldown_until_mono > 0 and time.monotonic() < self._cooldown_until_mono:
            return True
        if self._cooldown_until_mono > 0 and time.monotonic() >= self._cooldown_until_mono:
            self._cooldown_until_mono = 0.0
        if self.last_result_tick == 0:
            return False
        elapsed = current_tick - self.last_result_tick
        return elapsed < need
