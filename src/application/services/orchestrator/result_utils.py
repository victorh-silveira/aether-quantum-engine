"""Utilitarios para normalizacao de resultado de contratos."""


def api_settlement_label(status_str: str, profit: float) -> str:
    """Reduz status bruto da API e P&L para WIN, LOSS ou FLAT."""
    s = (status_str or "").upper()
    if s == "WON":
        return "WIN"
    if s == "LOST":
        return "LOSS"
    if s in {"EXPIRED", "SOLD"}:
        return "WIN" if profit > 0 else ("LOSS" if profit < 0 else "FLAT")
    if profit > 0:
        return "WIN"
    if profit < 0:
        return "LOSS"
    return "FLAT"
