"""Direcao de cluster inferida do Z-Score StatArb e correcao quant (M5 + StatArb)."""

from __future__ import annotations

from typing import Any

from src.application.services.llm.cluster_statarb_select import (
    resolve_statarb_cluster_config_for_tag,
    symbol_z_supports_direction,
)
from src.application.services.llm.macro_cluster_align import quant_trade_direction
from src.domain.models.trade import TradeDirection


def direction_from_statarb_z(
    z: float,
    *,
    hmm_state: int = 0,
    z_threshold: float = 2.5,
    min_abs_z: float = 0.0,
) -> TradeDirection | None:
    """Infere CALL/PUT a partir do Z-Score StatArb e regime HMM."""
    zf = float(z)
    floor = max(0.0, float(min_abs_z))
    if abs(zf) < floor:
        return None
    if int(hmm_state) == 1:
        return TradeDirection.CALL if zf >= floor else TradeDirection.PUT
    if zf <= -float(z_threshold):
        return TradeDirection.CALL
    if zf >= float(z_threshold):
        return TradeDirection.PUT
    return None


def _m5_implied_direction(metrics: dict[str, Any], target_sym: str) -> tuple[TradeDirection | None, str]:
    """Retorna direcao implied pelo M5 do simbolo e label micro (up/down)."""
    raw_map = metrics.get("index_m5_dir_by_symbol")
    if not isinstance(raw_map, dict):
        return None, ""
    micro = str(raw_map.get(target_sym) or "")
    if micro not in ("up", "down"):
        return None, ""
    implied = quant_trade_direction(micro)
    return implied, micro


def quant_direction_stack_enabled(corr_cfg: dict[str, Any] | None) -> bool:
    """True quando pilha M5/StatArb pode corrigir tag LLM do cluster."""
    c = corr_cfg if isinstance(corr_cfg, dict) else {}
    if "quant_direction_stack_enabled" in c:
        return bool(c.get("quant_direction_stack_enabled"))
    return bool(c.get("statarb_correct_llm_on_divergence", True))


def correct_cluster_direction_for_tag(
    direction: TradeDirection,
    *,
    macro_tag: str,
    target_sym: str,
    metrics: dict[str, Any],
    corr_cfg: dict[str, Any] | None,
    macro_cfg: dict[str, Any] | None,
) -> tuple[TradeDirection, bool, str]:
    """Aplica pilha M5 depois StatArb sobre direcao da tag LLM."""
    if not quant_direction_stack_enabled(corr_cfg):
        return direction, False, ""
    c = corr_cfg if isinstance(corr_cfg, dict) else {}

    m5_dir, micro = _m5_implied_direction(metrics, target_sym)
    if m5_dir is not None and m5_dir != direction:
        note = f"M5_DIR {direction.name}->{m5_dir.name} micro={micro}"
        return m5_dir, True, note

    spreads = metrics.get("statarb_spreads")
    if not isinstance(spreads, dict) or target_sym not in spreads:
        return direction, False, ""
    statarb_cfg = resolve_statarb_cluster_config_for_tag(c, macro_cfg, str(macro_tag or ""))
    z = float(spreads[target_sym])
    hmm_state = int(metrics.get("hmm_state", 0))
    z_threshold = float(statarb_cfg.get("z_threshold", 2.5))
    min_abs = float(statarb_cfg.get("min_abs_z", 0.0))
    if symbol_z_supports_direction(
        z,
        direction,
        hmm_state=hmm_state,
        z_threshold=z_threshold,
        min_abs_z=min_abs,
    ):
        return direction, False, ""
    implied = direction_from_statarb_z(
        z,
        hmm_state=hmm_state,
        z_threshold=z_threshold,
        min_abs_z=min_abs,
    )
    if implied is None or implied == direction:
        return direction, False, ""
    note = f"STATARB_DIR {direction.name}->{implied.name} z={z:.2f} hmm={hmm_state}"
    return implied, True, note


def correct_cluster_direction_for_divergence(
    direction: TradeDirection,
    *,
    macro_tag: str,
    target_sym: str,
    metrics: dict[str, Any],
    corr_cfg: dict[str, Any] | None,
    macro_cfg: dict[str, Any] | None,
) -> tuple[TradeDirection, bool, str]:
    """Wrapper de compatibilidade para correcao em tags de divergencia."""
    return correct_cluster_direction_for_tag(
        direction,
        macro_tag=macro_tag,
        target_sym=target_sym,
        metrics=metrics,
        corr_cfg=corr_cfg,
        macro_cfg=macro_cfg,
    )
