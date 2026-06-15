"""Cooldown de reentrada por simbolo apos loss recente."""


class SymbolLossCooldownMixin:
    """Rastreia simbolo da ultima loss e bloqueia reentrada por N ciclos ou velas."""

    last_loss_symbol: str | None
    last_loss_direction: str | None
    symbol_loss_cooldown: dict[str, int]
    symbol_loss_cooldown_unit: dict[str, str]

    def init_symbol_loss_cooldown(self) -> None:
        """Inicializa estado de cooldown pos-loss por simbolo."""
        self.last_loss_symbol = None
        self.last_loss_direction = None
        self.symbol_loss_cooldown = {}
        self.symbol_loss_cooldown_unit = {}

    def _resolve_symbol_loss_cooldown(self) -> tuple[int, str] | None:
        """Retorna duracao e unidade (candle|cycle) ou None se desativado."""
        candles = int(self.kelly_config.get("symbol_loss_cooldown_candles", 0))
        if candles > 0:
            return candles, "candle"
        cycles_raw = self.kelly_config.get("symbol_loss_cooldown_cycles")
        if cycles_raw is None:
            return None
        cycles = int(cycles_raw)
        if cycles <= 0:
            return None
        return cycles, "cycle"

    def tick_symbol_loss_cooldowns(self) -> None:
        """Decrementa cooldowns atrelados a fechamento de vela ancora."""
        self._tick_symbol_loss_cooldowns_for_unit("candle")

    def tick_symbol_loss_cycle_cooldowns(self) -> None:
        """Decrementa cooldowns atrelados a ciclos do orquestrador."""
        self._tick_symbol_loss_cooldowns_for_unit("cycle")

    def _tick_symbol_loss_cooldowns_for_unit(self, unit: str) -> None:
        """Decrementa entradas de cooldown da unidade informada."""
        for symbol in list(self.symbol_loss_cooldown.keys()):
            if self.symbol_loss_cooldown_unit.get(symbol) != unit:
                continue
            self.symbol_loss_cooldown[symbol] -= 1
            if self.symbol_loss_cooldown[symbol] <= 0:
                del self.symbol_loss_cooldown[symbol]
                self.symbol_loss_cooldown_unit.pop(symbol, None)

    def is_symbol_on_loss_cooldown(self, symbol: str) -> bool:
        """Indica se o simbolo ainda esta em cooldown apos loss recente."""
        return int(self.symbol_loss_cooldown.get(symbol, 0)) > 0

    def register_symbol_loss_cooldown(self, symbol: str, *, direction: str | None = None) -> None:
        """Marca simbolo com cooldown configuravel apos loss."""
        self.last_loss_symbol = symbol
        if direction:
            self.last_loss_direction = str(direction)
        resolved = self._resolve_symbol_loss_cooldown()
        if resolved is None:
            return
        cooldown, unit = resolved
        self.symbol_loss_cooldown[symbol] = cooldown
        self.symbol_loss_cooldown_unit[symbol] = unit

    def symbol_cooldown_state(self) -> dict:
        """Retorna campos de cooldown para persistencia."""
        return {
            "last_loss_symbol": self.last_loss_symbol,
            "last_loss_direction": self.last_loss_direction,
            "symbol_loss_cooldown": dict(self.symbol_loss_cooldown),
            "symbol_loss_cooldown_unit": dict(self.symbol_loss_cooldown_unit),
        }
