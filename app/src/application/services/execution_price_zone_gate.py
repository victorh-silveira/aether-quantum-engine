"""Gate de zona de preco: mean-reversion (BB/Keltner) com confirmacao de tendencia."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_price_zone_align import align_direction_to_price_zone
from src.application.services.execution_price_zone_meta import align_or_keep_meta_side
from src.application.services.execution_quality_gate_microstructure import resolve_skipped_cycles
from src.application.services.execution_quality_gate_starvation import starvation_decay_factor
from src.application.services.execution_runtime_config import resolve_price_zone_config as _resolve_price_zone
from src.application.services.execution_tcn_conviction import tcn_direction_lock_active, tcn_high_conviction_active
from src.domain.config_knobs import merge_settings_block
from src.domain.models.trade import TradeDirection


__all__ = (
    "ZONE_BUY",
    "ZONE_NONE",
    "ZONE_SELL",
    "align_direction_to_price_zone",
    "align_or_keep_meta_side",
    "apply_price_zone_gate",
    "apply_price_zone_gate_with_starvation",
    "direction_allowed_for_zone",
    "direction_for_price_zone",
    "resolve_price_zone",
    "resolve_price_zone_config",
    "zone_score",
)


ZONE_BUY = "BUY"
ZONE_SELL = "SELL"
ZONE_NONE = "NONE"


def resolve_price_zone_config(exec_cfg: dict[str, Any] | None) -> dict[str, Any]:
    """Extrai bloco orchestrator.execution.price_zone fail-closed."""
    cfg = _resolve_price_zone(exec_cfg if isinstance(exec_cfg, dict) else None)
    buy_max = float(cfg["buy_max"])
    sell_min = max(float(cfg["sell_min"]), buy_max)
    bb_w = max(0.0, float(cfg["bb_weight"]))
    kel_w = max(0.0, float(cfg["keltner_weight"]))
    total = bb_w + kel_w
    if total <= 0.0:
        ssot = merge_settings_block(("orchestrator", "execution", "price_zone"), None)
        bb_w = max(0.0, float(ssot["bb_weight"]))
        kel_w = max(0.0, float(ssot["keltner_weight"]))
        total = bb_w + kel_w
    if total <= 0.0:
        raise ValueError("price_zone.bb_weight/keltner_weight invalidos")
    return {
        "enabled": bool(cfg["enabled"]),
        "buy_max": buy_max,
        "sell_min": sell_min,
        "bb_weight": bb_w / total,
        "keltner_weight": kel_w / total,
        "neutral_mode": str(cfg.get("neutral_mode") or "reject"),
        "require_trend_agreement": bool(cfg["require_trend_agreement"]),
        "require_tcn_agreement": bool(cfg["require_tcn_agreement"]),
    }


def _metric_band(metrics: dict[str, Any], *keys: str, default: float = 0.5) -> float:
    """Le banda percentual clampada em [0, 1]."""
    for key in keys:
        raw = metrics.get(key)
        if raw is None and isinstance(metrics.get("indicators"), dict):
            raw = metrics["indicators"].get(key)
        if raw is None:
            continue
        try:
            return min(1.0, max(0.0, float(raw)))
        except (TypeError, ValueError):
            continue
    return default


def zone_score(metrics: dict[str, Any], cfg: dict[str, Any]) -> float:
    """Score blended bb_pct_b + keltner para localizacao na banda."""
    bb = _metric_band(metrics, "bb_pct_b", default=0.5)
    kel = _metric_band(metrics, "keltner", "keltner_pct_b", default=0.5)
    return float(cfg["bb_weight"]) * bb + float(cfg["keltner_weight"]) * kel


def resolve_price_zone(metrics: dict[str, Any], cfg: dict[str, Any] | None = None) -> str:
    """Retorna BUY, SELL ou NONE a partir do score de banda."""
    conf = cfg if isinstance(cfg, dict) else resolve_price_zone_config(None)
    if not bool(conf.get("enabled", False)):
        return ZONE_NONE
    score = zone_score(metrics, conf)
    if score <= float(conf["buy_max"]):
        return ZONE_BUY
    if score >= float(conf["sell_min"]):
        return ZONE_SELL
    if str(conf.get("neutral_mode") or "reject") == "nearest":
        return ZONE_BUY if score < 0.5 else ZONE_SELL
    return ZONE_NONE


def _trend_supports(direction: TradeDirection, metrics: dict[str, Any]) -> bool:
    """True quando tendencia/taxa vota a favor do lado."""
    trend = str(metrics.get("trend_direction") or "").upper()
    if trend in ("CALL", "PUT"):
        return trend == direction.name
    try:
        call_votes = int(metrics.get("call_votes", 0) or 0)
        put_votes = int(metrics.get("put_votes", 0) or 0)
    except (TypeError, ValueError):
        return False
    if call_votes == put_votes or direction not in (TradeDirection.CALL, TradeDirection.PUT):
        return False
    return (call_votes > put_votes) if direction == TradeDirection.CALL else (put_votes > call_votes)


def _tcn_direction(metrics: dict[str, Any], fallback: TradeDirection | None) -> TradeDirection | None:
    """Resolve lado TCN/DL em metrics ou fallback."""
    for key in ("dl_direction", "tcn_direction"):
        raw = str(metrics.get(key) or "").upper()
        if raw == TradeDirection.CALL.name:
            return TradeDirection.CALL
        if raw == TradeDirection.PUT.name:
            return TradeDirection.PUT
    return fallback


def direction_for_price_zone(zone: str) -> TradeDirection | None:
    """Lado mean-reversion da zona: BUY->CALL, SELL->PUT."""
    if zone == ZONE_BUY:
        return TradeDirection.CALL
    if zone == ZONE_SELL:
        return TradeDirection.PUT
    return None


def _zone_side_ok(zone: str, direction: TradeDirection) -> bool:
    """True quando o lado bate com a zona mean-reversion."""
    implied = direction_for_price_zone(zone)
    return implied is not None and direction == implied


def direction_allowed_for_zone(
    zone: str,
    direction: TradeDirection,
    metrics: dict[str, Any],
    cfg: dict[str, Any] | None = None,
    *,
    tcn_direction: TradeDirection | None = None,
) -> bool:
    """Valida lado da zona com tendencia/TCN opcionais."""
    conf = cfg if isinstance(cfg, dict) else resolve_price_zone_config(None)
    if not bool(conf.get("enabled", False)):
        return True
    implied = direction_for_price_zone(zone)
    if implied is None or direction != implied:
        return False
    if bool(conf.get("require_trend_agreement", False)) and not _trend_supports(direction, metrics):
        return False
    if not bool(conf.get("require_tcn_agreement", False)):
        return True
    tcn = _tcn_direction(metrics, tcn_direction)
    if tcn is not None and tcn == direction:
        return True
    if not tcn_direction_lock_active(metrics):
        metrics["price_zone_tcn_weak_defer"] = True
        return True
    return False


def _reject_reason(
    zone: str,
    direction: TradeDirection,
    metrics: dict[str, Any],
    conf: dict[str, Any],
    tcn_direction: TradeDirection | None,
) -> str:
    """Mapeia o primeiro motivo de rejeicao da zona."""
    implied = direction_for_price_zone(zone)
    if implied is None:
        return "price_zone_none"
    side = implied
    if bool(conf.get("require_trend_agreement", False)) and not _trend_supports(side, metrics):
        return "price_zone_trend_conflict"
    if bool(conf.get("require_tcn_agreement", False)):
        tcn = _tcn_direction(metrics, tcn_direction)
        if tcn is None or tcn != side:
            return "price_zone_tcn_conflict"
    if direction != side:
        return "price_zone_buy_requires_call" if zone == ZONE_BUY else "price_zone_sell_requires_put"
    return "price_zone_reject"


def _has_band_telemetry(metrics: dict[str, Any]) -> bool:
    """True quando metrics trazem bb/keltner para avaliar zona."""
    keys = ("bb_pct_b", "keltner", "keltner_pct_b")
    indicators = metrics.get("indicators") if isinstance(metrics.get("indicators"), dict) else {}
    for key in keys:
        if metrics.get(key) is not None:
            return True
        if indicators.get(key) is not None:
            return True
    return False


def apply_price_zone_gate(
    metrics: dict[str, Any],
    direction: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    tcn_direction: TradeDirection | None = None,
) -> str | None:
    """Aplica gate; NONE rejeita; BUY/SELL alinham o lado e so filtram se flags AND."""
    conf = resolve_price_zone_config(exec_cfg)
    if not bool(conf.get("enabled", False)):
        return None
    if not _has_band_telemetry(metrics):
        return None
    zone = resolve_price_zone(metrics, conf)
    metrics["price_zone"] = zone
    metrics["price_zone_score"] = zone_score(metrics, conf)
    if zone == ZONE_NONE:
        metrics.pop("price_zone_direction", None)
        return "price_zone_none"
    implied = TradeDirection.CALL if zone == ZONE_BUY else TradeDirection.PUT
    metrics["price_zone_direction"] = implied.name
    metrics["price_zone_aligned"] = direction != implied
    tcn = tcn_direction or direction
    if direction_allowed_for_zone(zone, implied, metrics, conf, tcn_direction=tcn):
        return None
    return _reject_reason(zone, implied, metrics, conf, tcn)


def apply_price_zone_gate_with_starvation(
    metrics: dict[str, Any],
    direction: TradeDirection,
    exec_cfg: dict[str, Any] | None,
    *,
    tcn_direction: TradeDirection | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
    force: bool = False,
) -> str | None:
    """Aplica price_zone; waiva sob starvation ou alta conviccao TCN."""
    if force:
        return None
    reason = apply_price_zone_gate(
        metrics,
        direction,
        exec_cfg,
        tcn_direction=tcn_direction,
    )
    if reason is None:
        return None
    if tcn_high_conviction_active(metrics):
        metrics["price_zone_conviction_waiver"] = True
        metrics["price_zone_waived_reason"] = str(reason)
        return None
    skipped = resolve_skipped_cycles(skipped_cycles_counter=skipped_cycles_counter, orch=orch)
    decay = starvation_decay_factor(
        skipped,
        exec_cfg=exec_cfg if isinstance(exec_cfg, dict) else None,
    )
    if decay + 1e-12 < 1.0:
        metrics["price_zone_starvation_waiver"] = True
        metrics["price_zone_waived_reason"] = str(reason)
        return None
    return reason
