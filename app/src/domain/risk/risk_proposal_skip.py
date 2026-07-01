"""Cooldown de simbolos apos falha de proposta na Deriv."""


class ProposalSkipMixin:
    """Mixin com controle de skip temporario por simbolo."""

    proposal_skip_cycles: dict[str, int]

    def register_proposal_failure(self, symbol: str, *, cycles: int = 6) -> None:
        """Marca simbolo para pular selecao apos falha de proposta na Deriv."""
        hold = max(int(cycles), int(self.proposal_skip_cycles.get(str(symbol), 0)))
        self.proposal_skip_cycles[str(symbol)] = hold

    def decay_proposal_skip_cycles(self) -> None:
        """Reduz cooldown de simbolos com proposta rejeitada."""
        expired: list[str] = []
        for symbol, remaining in self.proposal_skip_cycles.items():
            left = int(remaining) - 1
            if left <= 0:
                expired.append(symbol)
            else:
                self.proposal_skip_cycles[symbol] = left
        for symbol in expired:
            self.proposal_skip_cycles.pop(symbol, None)

    def proposal_skip_symbols(self) -> frozenset[str]:
        """Simbolos temporariamente excluidos apos falha de proposta."""
        return frozenset(s for s, n in self.proposal_skip_cycles.items() if int(n) > 0)
