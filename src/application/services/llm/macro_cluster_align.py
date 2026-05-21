"""Alinhamento de tags US_CLUSTER/EU_CLUSTER ao voto quantitativo macro."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.macro_config import MacroSnapshot, resolve_macro_config


def cluster_trade_direction(cluster_dir: str) -> str | None:
    """Converte direcao de cluster em CALL/PUT para indices RISE_FALL."""
    if cluster_dir == "up":
        return "CALL"
    if cluster_dir == "down":
        return "PUT"
    return None


def expected_cluster_tags_line(
    *,
    tag: str,
    us_dir: str,
    eu_dir: str,
    us_strength: float,
    eu_strength: float,
    macro_cfg: dict[str, Any] | None = None,
) -> str:
    """Linha de referencia CALL/PUT por cluster derivada do voto quantitativo."""
    cfg = resolve_macro_config(macro_cfg)
    floor = float(cfg["confluence_conviction_floor"])
    if tag == "risk_on" and min(us_strength, eu_strength) >= floor:
        return "CLUSTER_QUANT_OBRIGATORIO: US_CLUSTER=CALL | EU_CLUSTER=CALL"
    if tag == "risk_off" and min(us_strength, eu_strength) >= floor:
        return "CLUSTER_QUANT_OBRIGATORIO: US_CLUSTER=PUT | EU_CLUSTER=PUT"
    us_tok = cluster_trade_direction(us_dir)
    eu_tok = cluster_trade_direction(eu_dir)
    if us_tok and us_strength >= floor and eu_tok and eu_strength >= floor:
        return f"CLUSTER_QUANT_OBRIGATORIO: US_CLUSTER={us_tok} | EU_CLUSTER={eu_tok}"
    if us_tok and us_strength >= floor:
        return f"CLUSTER_QUANT_US: US_CLUSTER={us_tok}"
    if eu_tok and eu_strength >= floor:
        return f"CLUSTER_QUANT_EU: EU_CLUSTER={eu_tok}"
    return ""


def reconcile_cluster_tags_with_macro(
    us_tag: str | None,
    eu_tag: str | None,
    snapshot: MacroSnapshot,
    macro_cfg: dict[str, Any] | None = None,
) -> tuple[str | None, str | None, bool, str]:
    """Alinha US_CLUSTER e EU_CLUSTER ao voto quantitativo quando a LLM diverge do macro."""
    cfg = resolve_macro_config(macro_cfg)
    if not cfg["align_clusters_with_macro_vote"]:
        return us_tag, eu_tag, False, ""

    floor = float(cfg["confluence_conviction_floor"])
    tag = snapshot.tag
    out_us = us_tag if us_tag in ("CALL", "PUT") else None
    out_eu = eu_tag if eu_tag in ("CALL", "PUT") else None
    changed = False

    if tag == "risk_on" and min(snapshot.us_strength, snapshot.eu_strength) >= floor:
        target = "CALL"
    elif tag == "risk_off" and min(snapshot.us_strength, snapshot.eu_strength) >= floor:
        target = "PUT"
    elif tag.startswith("divergence"):
        target = None
    else:
        return out_us, out_eu, False, ""

    if target is not None:
        if out_us != target:
            out_us = target
            changed = True
        if out_eu != target:
            out_eu = target
            changed = True
    else:
        q_us = cluster_trade_direction(snapshot.us_dir)
        q_eu = cluster_trade_direction(snapshot.eu_dir)
        if q_us and snapshot.us_strength >= floor and out_us != q_us:
            out_us = q_us
            changed = True
        if q_eu and snapshot.eu_strength >= floor and out_eu != q_eu:
            out_eu = q_eu
            changed = True

    note = f"MACRO_CLUSTER_ALIGN tag={tag}" if changed else ""
    return out_us, out_eu, changed, note
