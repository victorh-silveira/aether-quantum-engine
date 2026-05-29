"""Referencias quantitativas de cluster para o prompt Medallion."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.macro_config import resolve_macro_config
from src.domain.models.trade import TradeDirection


def cluster_trade_direction(cluster_dir: str) -> str | None:
    """Converte direcao de cluster em CALL/PUT para indices RISE_FALL."""
    if cluster_dir == "up":
        return "CALL"
    if cluster_dir == "down":
        return "PUT"
    return None


def quant_trade_direction(cluster_dir: str) -> TradeDirection | None:
    tok = cluster_trade_direction(cluster_dir)
    if tok == "CALL":
        return TradeDirection.CALL
    if tok == "PUT":
        return TradeDirection.PUT
    return None


def align_cluster_dirs_for_divergence_tag(
    macro_tag: str,
    *,
    us_dir_quant: str,
    eu_dir_quant: str,
    us_dir: TradeDirection | None,
    eu_dir: TradeDirection | None,
) -> tuple[TradeDirection | None, TradeDirection | None]:
    if macro_tag == "divergence_us_leads":
        leader = quant_trade_direction(us_dir_quant)
        if us_dir is None and leader is not None:
            return leader, eu_dir
        return us_dir, eu_dir
    if macro_tag == "divergence_eu_leads":
        leader = quant_trade_direction(eu_dir_quant)
        if eu_dir is None and leader is not None:
            return us_dir, leader
        return us_dir, eu_dir
    return us_dir, eu_dir


def expected_cluster_tags_line(
    *,
    tag: str,
    us_dir: str,
    eu_dir: str,
    us_strength: float,
    eu_strength: float,
    macro_cfg: dict[str, Any] | None = None,
) -> str:
    """Linha informativa CALL/PUT por cluster derivada do voto quantitativo."""
    cfg = resolve_macro_config(macro_cfg)
    floor = float(cfg["confluence_conviction_floor"])
    if tag == "risk_on" and min(us_strength, eu_strength) >= floor:
        return "CLUSTER_QUANT_REF: US_CLUSTER=CALL | EU_CLUSTER=CALL"
    if tag == "risk_off" and min(us_strength, eu_strength) >= floor:
        return "CLUSTER_QUANT_REF: US_CLUSTER=PUT | EU_CLUSTER=PUT"
    us_tok = cluster_trade_direction(us_dir)
    eu_tok = cluster_trade_direction(eu_dir)
    if us_tok and us_strength >= floor and eu_tok and eu_strength >= floor:
        return f"CLUSTER_QUANT_REF: US_CLUSTER={us_tok} | EU_CLUSTER={eu_tok}"
    if us_tok and us_strength >= floor:
        return f"CLUSTER_QUANT_REF: US_CLUSTER={us_tok}"
    if eu_tok and eu_strength >= floor:
        return f"CLUSTER_QUANT_REF: EU_CLUSTER={eu_tok}"
    return ""
