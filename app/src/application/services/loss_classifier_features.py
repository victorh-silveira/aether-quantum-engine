"""Vetor tabular 24D para o loss-classifier HTTP."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


LOSS_FEATURE_DIM = 24


def _f(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Le float seguro de metrics com fallback."""
    raw = metrics.get(key)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _clip01(value: float) -> float:
    """Projeta valor no intervalo [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _clip3(value: float) -> float:
    """Projeta valor no intervalo [-3, 3]."""
    return max(-3.0, min(3.0, float(value)))


def _tick_accel(metrics: dict[str, Any]) -> float:
    """Le aceleracao de ticks do bloco flow_features, clipada."""
    flow = metrics.get("flow_features")
    if isinstance(flow, dict) and flow.get("micro_tick_acceleration") is not None:
        try:
            return _clip3(float(flow["micro_tick_acceleration"]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def build_loss_feature_vector(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    pending: float = 0.0,
    linear: int = 0,
    bankroll: float = 0.0,
) -> list[float]:
    """Monta vetor 24D a partir de telemetria SCALE/Kelly/Cal."""
    regime = str(metrics.get("scale_micro_regime") or "chop").lower()
    explos = 1.0 if regime == "explosion" else 0.0
    retract = 1.0 if regime == "retraction" else 0.0
    chop = 1.0 if regime not in {"explosion", "retraction"} else 0.0
    tcn = str(metrics.get("tcn_direction") or metrics.get("dl_direction") or "").upper()
    tape = str(metrics.get("scale_tape_consensus") or "").upper()
    tape_vs = 1.0 if tcn and tape and tcn != tape else 0.0
    mini_prev = str(metrics.get("scale_mini_prev_bar_dir") or "").upper()
    mini_cur = str(metrics.get("scale_mini_bar_dir") or "").upper()
    mini_oppose = (
        1.0
        if mini_prev in {TradeDirection.CALL.name, TradeDirection.PUT.name}
        and mini_cur == mini_prev
        and mini_prev != exec_dir.name
        else 0.0
    )
    pending_norm = _clip01(float(pending) / float(bankroll)) if bankroll > 1e-9 else _clip01(float(pending) / 1000.0)
    raw_edge = metrics.get("predicted_payoff_edge")
    if raw_edge is None:
        edge = _f(metrics, "edge_zscore")
    else:
        try:
            edge = float(raw_edge)
        except (TypeError, ValueError):
            edge = _f(metrics, "edge_zscore")
    vector = [
        _clip01(_f(metrics, "direction_margin")),
        _clip01(_f(metrics, "calibrated_prob", _f(metrics, "raw_prob", 0.5))),
        1.0 if bool(metrics.get("scale_adapted")) else 0.0,
        1.0 if bool(metrics.get("scale_discordance")) else 0.0,
        explos,
        retract,
        chop,
        tape_vs,
        _clip01(float(linear) / 5.0),
        pending_norm,
        _clip3(edge),
        _clip01(_f(metrics, "conviction", _f(metrics, "trade_score", 0.55))),
        _clip01(_f(metrics, "live_n") / 40.0),
        _clip01(_f(metrics, "live_wr")),
        1.0 if bool(metrics.get("scale_tape_strong")) else 0.0,
        1.0 if bool(metrics.get("scale_mili_oppose_tcn")) else 0.0,
        1.0 if exec_dir.name == TradeDirection.PUT.name else 0.0,
        1.0 if tcn == TradeDirection.CALL.name else 0.0,
        _clip01(_f(metrics, "raw_prob", 0.5)),
        _tick_accel(metrics),
        _clip01(_f(metrics, "val_accuracy", _f(metrics, "acc", 0.55))),
        _clip3(_f(metrics, "edge_zscore")),
        _clip3(_f(metrics, "bb_width_z", _f(metrics, "bbw", 0.0))),
        mini_oppose,
    ]
    if len(vector) != LOSS_FEATURE_DIM:
        raise ValueError(f"loss feature dim {len(vector)} != {LOSS_FEATURE_DIM}")
    return vector
