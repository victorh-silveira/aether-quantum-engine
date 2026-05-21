"""Linhas de contexto FX de referencia para confluencia macro."""

from __future__ import annotations

from typing import Any


def fx_reference_context_line(tag: str, fx_pairs: dict[str, Any] | None = None) -> str:
    """Monta linha de contexto FX de referencia (nao negociavel) para o prompt."""
    pairs = fx_pairs if isinstance(fx_pairs, dict) else {}
    usdjpy = pairs.get("usdjpy") if isinstance(pairs.get("usdjpy"), dict) else {}
    aud = pairs.get("audusd") if isinstance(pairs.get("audusd"), dict) else {}
    nzd = pairs.get("nzdusd") if isinstance(pairs.get("nzdusd"), dict) else {}

    if tag == "risk_on":
        jpy = str(usdjpy.get("risk_on", "RISE")).upper()
        aud_m = str(aud.get("risk_on", "RISE")).upper()
        nzd_m = str(nzd.get("risk_on", "RISE")).upper()
        mood = "Risk-On"
    elif tag == "risk_off":
        jpy = str(usdjpy.get("risk_off", "FALL")).upper()
        aud_m = str(aud.get("risk_off", "FALL")).upper()
        nzd_m = str(nzd.get("risk_off", "FALL")).upper()
        mood = "Risk-Off"
    elif tag == "divergence_us_leads":
        mood = "Divergencia US lidera EU"
        jpy = "RISE"
        aud_m = "RISE"
        nzd_m = "RISE"
    elif tag == "divergence_eu_leads":
        mood = "Divergencia EU lidera US"
        jpy = "FALL"
        aud_m = "RISE"
        nzd_m = "RISE"
    else:
        mood = "Indefinido"
        jpy = "FLAT"
        aud_m = "FLAT"
        nzd_m = "FLAT"

    return (
        f"CONTEXTO_FX_REF: {mood} | USDJPY {jpy} | AUDUSD {aud_m} | NZDUSD {nzd_m} "
        "(referencia macro, sem execucao nestes pares)"
    )
