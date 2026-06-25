"""Cooldown de reentrada por simbolo apos loss recente (desativado)."""


class SymbolLossCooldownMixin:
    """Rastreia simbolo da ultima loss (para fins de diversificacao) mas sem cooldowns ativos."""

    last_loss_symbol: str | None
    last_loss_direction: str | None
    symbol_loss_cooldown: dict[str, int]
    symbol_loss_cooldown_unit: dict[str, str]

    def init_symbol_loss_cooldown(self) -> None:
        """Inicializa estado de cooldown (vazio)."""
        self.last_loss_symbol = None
        self.last_loss_direction = None
        self.symbol_loss_cooldown = {}
        self.symbol_loss_cooldown_unit = {}

    def tick_symbol_loss_cooldowns(self) -> None:
        """No-op."""
        pass

    def tick_symbol_loss_cycle_cooldowns(self) -> None:
        """No-op."""
        pass

    def is_symbol_on_loss_cooldown(self, symbol: str) -> bool:
        """Sempre False (desativado)."""
        _ = symbol
        return False

    def register_symbol_loss_cooldown(self, symbol: str, *, direction: str | None = None) -> None:
        """Apenas registra o ultimo simbolo/direcao que sofreu loss, sem aplicar cooldown."""
        self.last_loss_symbol = symbol
        if direction:
            self.last_loss_direction = str(direction)
        self.symbol_loss_cooldown = {}
        self.symbol_loss_cooldown_unit = {}

    def symbol_cooldown_state(self) -> dict:
        """Retorna campos de estado (vazio)."""
        return {
            "last_loss_symbol": self.last_loss_symbol,
            "last_loss_direction": self.last_loss_direction,
            "symbol_loss_cooldown": {},
            "symbol_loss_cooldown_unit": {},
        }
