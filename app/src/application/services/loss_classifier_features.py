"""Vetor tabular 24D para o loss-classifier HTTP."""

from __future__ import annotations

from typing import Any

from src.application.services.ml_schema_hash import canonical_schema_hash
from src.domain.models.trade import TradeDirection


LOSS_FEATURE_DIM = 24
LOSS_FEATURE_NAMES: tuple[str, ...] = (
    "direction_margin",
    "calibrated_prob",
    "cal_raw_discord",
    "scale_discordance",
    "regime_explosion",
    "regime_retraction",
    "regime_chop",
    "tape_vs_tcn",
    "linear_norm",
    "pending_norm",
    "predicted_payoff_edge",
    "conviction",
    "live_n_norm",
    "hurst",
    "scale_tape_strong",
    "scale_mili_oppose_tcn",
    "adx",
    "variance_ratio",
    "raw_prob",
    "tick_accel",
    "val_accuracy",
    "keltner_deviation",
    "bb_width_z",
    "mini_oppose",
)


def loss_feature_schema_hash() -> str:
    """Hash do schema 24D enviado ao sidecar de loss."""
    return canonical_schema_hash(LOSS_FEATURE_NAMES)


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


def _ind(metrics: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Le indicador do bucket indicators ou da raiz de metrics."""
    chunk = metrics.get("indicators")
    if isinstance(chunk, dict) and chunk.get(key) is not None:
        return _f(chunk, key, default)
    return _f(metrics, key, default)


def _tick_accel(metrics: dict[str, Any]) -> float:
    """Le aceleracao de ticks do bloco flow_features, clipada."""
    flow = metrics.get("flow_features")
    if isinstance(flow, dict) and flow.get("micro_tick_acceleration") is not None:
        try:
            return _clip3(float(flow["micro_tick_acceleration"]))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _keltner_dev(metrics: dict[str, Any]) -> float:
    """Desvio Keltner do flow ou keltner_pct_b centrado."""
    flow = metrics.get("flow_features")
    if isinstance(flow, dict) and flow.get("keltner_deviation_ratio") is not None:
        try:
            return _clip3(float(flow["keltner_deviation_ratio"]))
        except (TypeError, ValueError):
            pass
    return _clip3(_ind(metrics, "keltner", 0.5) - 0.5)


def _payoff_edge(metrics: dict[str, Any]) -> float:
    """Edge continuo do meta; 0 se ausente (nao duplica edge_zscore)."""
    raw_edge = metrics.get("predicted_payoff_edge")
    if raw_edge is None:
        return 0.0
    try:
        return _clip3(float(raw_edge))
    except (TypeError, ValueError):
        return 0.0


def build_loss_feature_vector(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    pending: float = 0.0,
    linear: int = 0,
    bankroll: float = 0.0,
) -> list[float]:
    """Monta vetor 24D causal (Cal/raw, regime, meta edge, Hurst/ADX/VR/Keltner)."""
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
    cal = _clip01(_f(metrics, "calibrated_prob", _f(metrics, "raw_prob", 0.5)))
    raw = _clip01(_f(metrics, "raw_prob", 0.5))
    vector = [
        _clip01(_f(metrics, "direction_margin")),
        cal,
        _clip01(abs(cal - raw)),
        1.0 if bool(metrics.get("scale_discordance")) else 0.0,
        explos,
        retract,
        chop,
        tape_vs,
        _clip01(float(linear) / 5.0),
        pending_norm,
        _payoff_edge(metrics),
        _clip01(_f(metrics, "conviction", _f(metrics, "trade_score", 0.55))),
        _clip01(_f(metrics, "live_n") / 40.0),
        _clip01(_ind(metrics, "hurst", 0.5)),
        1.0 if bool(metrics.get("scale_tape_strong")) else 0.0,
        1.0 if bool(metrics.get("scale_mili_oppose_tcn")) else 0.0,
        _clip01(_ind(metrics, "adx", 0.0)),
        _clip3(_ind(metrics, "variance_ratio", 1.0)),
        raw,
        _tick_accel(metrics),
        _clip01(_f(metrics, "val_accuracy", _f(metrics, "acc", 0.55))),
        _keltner_dev(metrics),
        _clip3(_f(metrics, "bb_width_z", _f(metrics, "bbw", _ind(metrics, "bb_width", 0.0)))),
        mini_oppose,
    ]
    if len(vector) != LOSS_FEATURE_DIM:
        raise ValueError(f"loss feature dim {len(vector)} != {LOSS_FEATURE_DIM} names={len(LOSS_FEATURE_NAMES)}")
    return vector
