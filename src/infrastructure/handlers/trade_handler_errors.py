"""Erros de compra cuja confirmacao pode ter se perdido no transporte."""


class PurchaseOutcomeUnknownError(RuntimeError):
    """O broker pode ter aceitado a compra apesar da falha de resposta."""
