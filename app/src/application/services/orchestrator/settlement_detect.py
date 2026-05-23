"""Deteccao de contrato liquidado a partir de payloads da API Deriv."""


def contract_payload_is_settled(c: dict) -> bool:
    """Retorna True quando o payload indica contrato encerrado na corretora."""
    if int(c.get("is_settled") or 0) == 1:
        return True
    if int(c.get("is_expired") or 0) == 1:
        return True
    status = (c.get("status") or "").upper()
    return status in ("WON", "LOST", "EXPIRED", "SOLD", "CLOSED")
