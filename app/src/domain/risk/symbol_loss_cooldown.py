"""Cooldown de reentrada por simbolo apos loss recente."""


class SymbolLossCooldownMixin:
    """Rastreia simbolo da ultima loss e bloqueia reentrada por N ciclos."""

    last_loss_symbol: str | None
    last_loss_direction: str | None
    symbol_loss_cooldown: dict[str, int]

    def init_symbol_loss_cooldown(self) -> None:
        """Inicializa estado de cooldown pos-loss por simbolo."""
        self.last_loss_symbol = None
        self.last_loss_direction = None
        self.symbol_loss_cooldown = {}

    def tick_symbol_loss_cooldowns(self) -> None:
        """Decrementa cooldown de reentrada apos loss por simbolo."""
        for symbol in list(self.symbol_loss_cooldown.keys()):
            self.symbol_loss_cooldown[symbol] -= 1
            if self.symbol_loss_cooldown[symbol] <= 0:
                del self.symbol_loss_cooldown[symbol]

    def is_symbol_on_loss_cooldown(self, symbol: str) -> bool:
        """Indica se o simbolo ainda esta em cooldown apos loss recente."""
        return int(self.symbol_loss_cooldown.get(symbol, 0)) > 0

    def register_symbol_loss_cooldown(self, symbol: str, *, direction: str | None = None) -> None:
        """Marca simbolo com cooldown configuravel apos loss."""
        self.last_loss_symbol = symbol
        if direction:
            self.last_loss_direction = str(direction)
        cooldown = int(self.kelly_config.get("symbol_loss_cooldown_candles", 0))
        if cooldown <= 0:
            cooldown = int(self.kelly_config.get("symbol_loss_cooldown_cycles", 2))
        if cooldown > 0:
            self.symbol_loss_cooldown[symbol] = cooldown

    def symbol_cooldown_state(self) -> dict:
        """Retorna campos de cooldown para persistencia."""
        return {
            "last_loss_symbol": self.last_loss_symbol,
            "last_loss_direction": self.last_loss_direction,
            "symbol_loss_cooldown": dict(self.symbol_loss_cooldown),
        }
