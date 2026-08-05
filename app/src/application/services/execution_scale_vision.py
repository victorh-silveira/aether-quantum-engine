"""Visao multi-escala MACRO/MICRO/MINI/MILI para telemetria e concordancia soft."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

import numpy as np

from src.application.services.execution_scale_micro import classify_micro_regime, micro_regime_token
from src.application.services.execution_scale_tape import (
    bar_direction_at,
    compute_tape_strong,
    last_bar_direction,
    mini_bar_pair_agrees,
    mini_pair_opposes_tcn,
    prev_bar_direction,
    tape_consensus,
)
from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_int, require_keys
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings


_SCALE_VISION_KEYS = (
    "enabled",
    "slope_bars",
    "kelly_mult_discord",
    "min_disagree_to_dampen",
    "block_recover_on_discord",
    "use_last_bar",
    "adapt_direction_enabled",
    "adapt_require_raw_extreme",
    "adapt_require_bar_pair_agree",
    "adapt_allow_strong_tape",
    "adapt_strong_mini_pair",
    "adapt_kelly_p_floor",
    "adapt_min_votes",
    "adapt_on_retraction",
    "adapt_on_explosion",
    "adapt_on_mili_tape",
    "retraction_require_mili",
    "retraction_use_tick_accel",
    "max_stake_pct_discord",
)

__all__ = (
    "bar_direction_at",
    "compute_scale_directions",
    "compute_tape_strong",
    "format_scale_audit_line",
    "format_scale_ind_token",
    "last_bar_direction",
    "mili_direction_from_flow",
    "mini_bar_pair_agrees",
    "parse_scale_vision_config",
    "prev_bar_direction",
    "slope_direction",
    "tape_consensus",
)


def parse_scale_vision_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve orchestrator.execution.scale_vision com merge SSOT."""
    block = require_keys(
        merge_settings_block(
            ("orchestrator", "execution", "scale_vision"),
            raw if isinstance(raw, dict) else None,
        ),
        _SCALE_VISION_KEYS,
        "orchestrator.execution.scale_vision",
    )
    return {
        "enabled": require_bool(block, "enabled"),
        "slope_bars": max(2, require_int(block, "slope_bars")),
        "kelly_mult_discord": max(0.05, min(1.0, require_float(block, "kelly_mult_discord"))),
        "min_disagree_to_dampen": max(1, require_int(block, "min_disagree_to_dampen")),
        "block_recover_on_discord": require_bool(block, "block_recover_on_discord"),
        "use_last_bar": require_bool(block, "use_last_bar"),
        "adapt_direction_enabled": require_bool(block, "adapt_direction_enabled"),
        "adapt_require_raw_extreme": require_bool(block, "adapt_require_raw_extreme"),
        "adapt_require_bar_pair_agree": require_bool(block, "adapt_require_bar_pair_agree"),
        "adapt_allow_strong_tape": require_bool(block, "adapt_allow_strong_tape"),
        "adapt_strong_mini_pair": require_bool(block, "adapt_strong_mini_pair"),
        "adapt_kelly_p_floor": float(load_kelly_runtime_from_settings()["kelly_p_floor"]),
        "adapt_min_votes": max(1, require_int(block, "adapt_min_votes")),
        "adapt_on_retraction": require_bool(block, "adapt_on_retraction"),
        "adapt_on_explosion": require_bool(block, "adapt_on_explosion"),
        "adapt_on_mili_tape": require_bool(block, "adapt_on_mili_tape"),
        "retraction_require_mili": require_bool(block, "retraction_require_mili"),
        "retraction_use_tick_accel": require_bool(block, "retraction_use_tick_accel"),
        "max_stake_pct_discord": max(0.0, min(0.05, require_float(block, "max_stake_pct_discord"))),
    }


def slope_direction(closes: np.ndarray | list[float], *, bars: int = 5) -> str | None:
    """Direcao CALL/PUT por slope dos closes recentes; None se dados insuficientes."""
    arr = np.asarray(closes, dtype=np.float64).reshape(-1)
    n = max(2, int(bars))
    if arr.size < n:
        return None
    window = arr[-n:]
    delta = float(window[-1] - window[0])
    if abs(delta) <= 1e-12:
        return None
    return TradeDirection.CALL.name if delta > 0.0 else TradeDirection.PUT.name


def mili_direction_from_flow(flow: dict[str, Any] | None, tick_buffer: Any | None, symbol: str) -> str | None:
    """Direcao MILI a partir de velocity/acceleration de ticks."""
    vel = 0.0
    accel = 0.0
    if isinstance(flow, dict):
        try:
            vel = float(flow.get("price_velocity") or flow.get("micro_tick_velocity") or 0.0)
        except (TypeError, ValueError):
            vel = 0.0
        try:
            accel = float(flow.get("micro_tick_acceleration") or flow.get("price_acceleration") or 0.0)
        except (TypeError, ValueError):
            accel = 0.0
    if tick_buffer is not None and hasattr(tick_buffer, "live_tick_acceleration"):
        with suppress(Exception):
            accel = float(tick_buffer.live_tick_acceleration(str(symbol)))
    score = vel + 0.5 * accel
    if abs(score) <= 1e-12:
        return None
    return TradeDirection.CALL.name if score > 0.0 else TradeDirection.PUT.name


def _closes_from_stream(stream: Any, getter: str, symbol: str) -> np.ndarray:
    """Le serie close do stream via getter nomeado."""
    return _field_from_stream(stream, getter, symbol, "close")


def _field_from_stream(stream: Any, getter: str, symbol: str, field: str) -> np.ndarray:
    """Le serie OHLC do stream via getter nomeado."""
    fn = getattr(stream, getter, None)
    if not callable(fn):
        return np.asarray([], dtype=np.float64)
    arr = fn(str(symbol), field)
    if arr is None:
        return np.asarray([], dtype=np.float64)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _seed_scale_metrics(metrics: dict[str, Any], micro_name: str | None) -> None:
    """Inicializa campos SCALE no metrics."""
    metrics["scale_micro_dir"] = micro_name
    metrics["scale_macro_dir"] = None
    metrics["scale_mini_dir"] = None
    metrics["scale_mili_dir"] = None
    metrics["scale_mini_bar_dir"] = None
    metrics["scale_mini_prev_bar_dir"] = None
    metrics["scale_micro_bar_dir"] = None
    metrics["scale_micro_prev_bar_dir"] = None
    metrics["scale_tape_consensus"] = None
    metrics["scale_tape_strong"] = False
    metrics["scale_mini_pair_oppose"] = False
    metrics["scale_micro_regime"] = "chop"
    metrics["scale_micro_side"] = None
    metrics["scale_retraction_vs_tcn"] = False
    metrics["scale_mili_oppose_tcn"] = False
    metrics["scale_agree_n"] = 0
    metrics["scale_disagree_n"] = 0
    metrics["scale_discordance"] = False


def compute_scale_directions(
    orch: Any | None,
    symbol: str | None,
    tcn_dir: TradeDirection | str | None,
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Anexa dirs MACRO/MICRO/MINI/MILI, last-bar e consenso de fita ao metrics."""
    vision = cfg if isinstance(cfg, dict) else parse_scale_vision_config(None)
    micro_name = (
        tcn_dir.name
        if isinstance(tcn_dir, TradeDirection)
        else str(tcn_dir or metrics.get("exec_direction") or metrics.get("resolved_direction") or "").upper()
    )
    if micro_name not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        micro_name = None
    _seed_scale_metrics(metrics, micro_name)
    if not bool(vision.get("enabled", True)) or orch is None or not symbol:
        metrics["scale_reason"] = "disabled" if not bool(vision.get("enabled", True)) else "no_orch"
        return metrics
    stream = getattr(orch, "stream", None)
    slope_bars = int(vision.get("slope_bars", 5))
    use_last_bar = bool(vision.get("use_last_bar", True))
    if stream is not None:
        metrics["scale_macro_dir"] = slope_direction(
            _closes_from_stream(stream, "get_numpy_series", str(symbol)), bars=slope_bars
        )
        mini_closes = _closes_from_stream(stream, "get_mini_numpy_series", str(symbol))
        metrics["scale_mini_dir"] = slope_direction(mini_closes, bars=slope_bars)
        if use_last_bar:
            mini_opens = _field_from_stream(stream, "get_mini_numpy_series", str(symbol), "open")
            metrics["scale_mini_bar_dir"] = last_bar_direction(mini_opens, mini_closes)
            metrics["scale_mini_prev_bar_dir"] = prev_bar_direction(mini_opens, mini_closes)
            micro_closes = _closes_from_stream(stream, "get_micro_numpy_series", str(symbol))
            micro_opens = _field_from_stream(stream, "get_micro_numpy_series", str(symbol), "open")
            metrics["scale_micro_bar_dir"] = last_bar_direction(micro_opens, micro_closes)
            metrics["scale_micro_prev_bar_dir"] = prev_bar_direction(micro_opens, micro_closes)
    flow = metrics.get("flow_features") if isinstance(metrics.get("flow_features"), dict) else None
    tick_buffer = getattr(stream, "tick_buffer", None) if stream is not None else None
    metrics["scale_mili_dir"] = mili_direction_from_flow(flow, tick_buffer, str(symbol))
    mini_curr = metrics["scale_mini_bar_dir"] if use_last_bar else None
    mini_prev = metrics["scale_mini_prev_bar_dir"] if use_last_bar else None
    mini_peer = mini_curr if mini_curr is not None else metrics["scale_mini_dir"]
    peers = [metrics["scale_macro_dir"], mini_prev, mini_peer, metrics["scale_mili_dir"]]
    agree = 0
    disagree = 0
    if micro_name:
        agree += 1
        for peer in peers:
            if peer is None:
                continue
            if peer == micro_name:
                agree += 1
            else:
                disagree += 1
    metrics["scale_agree_n"] = int(agree)
    metrics["scale_disagree_n"] = int(disagree)
    min_disagree = int(vision.get("min_disagree_to_dampen", 2))
    metrics["scale_discordance"] = bool(micro_name and disagree >= min_disagree)
    if micro_name and mini_pair_opposes_tcn(metrics, micro_name):
        metrics["scale_discordance"] = True
        metrics["scale_mini_pair_oppose"] = True
    else:
        metrics["scale_mini_pair_oppose"] = False
    adapt_votes = int(vision.get("adapt_min_votes", 2))
    consensus = tape_consensus(
        [
            metrics["scale_mini_prev_bar_dir"],
            metrics["scale_mini_bar_dir"],
            metrics["scale_micro_prev_bar_dir"],
            metrics["scale_micro_bar_dir"],
            metrics["scale_mili_dir"],
        ],
        min_votes=adapt_votes,
    )
    metrics["scale_tape_consensus"] = consensus
    metrics["scale_tape_strong"] = compute_tape_strong(
        metrics,
        consensus,
        mini_pair_sufficient=bool(vision.get("adapt_strong_mini_pair", True)),
    )
    classify_micro_regime(metrics, micro_name, cfg=vision)
    metrics["scale_reason"] = "discord" if metrics["scale_discordance"] else "ok"
    return metrics


def format_scale_audit_line(metrics: dict[str, Any] | None) -> str:
    """Linha SCALE para log IND/CLUSTER."""
    m = metrics if isinstance(metrics, dict) else {}
    adapted = 1 if bool(m.get("scale_adapted")) else 0
    micro = micro_regime_token(m.get("scale_micro_regime"))
    return (
        f"SCALE || MACRO={m.get('scale_macro_dir') or '-'} "
        f"MICRO={m.get('scale_micro_dir') or '-'} "
        f"MINI={m.get('scale_mini_dir') or '-'} "
        f"MILI={m.get('scale_mili_dir') or '-'} "
        f"mi_prev={m.get('scale_mini_prev_bar_dir') or '-'} "
        f"mi_cur={m.get('scale_mini_bar_dir') or '-'} "
        f"mc_prev={m.get('scale_micro_prev_bar_dir') or '-'} "
        f"mc_cur={m.get('scale_micro_bar_dir') or '-'} "
        f"tape={m.get('scale_tape_consensus') or '-'} "
        f"micro={micro} "
        f"agree={int(m.get('scale_agree_n') or 0)}/4 "
        f"discord={bool(m.get('scale_discordance'))} "
        f"adapted={adapted}"
    )


def format_scale_ind_token(metrics: dict[str, Any] | None) -> str:
    """Token condensado SCALE para linha IND."""
    m = metrics if isinstance(metrics, dict) else {}
    tcn = m.get("tcn_direction") or m.get("scale_micro_dir") or "-"
    tape = m.get("scale_tape_consensus") or "-"
    adapted = 1 if bool(m.get("scale_adapted")) else 0
    mi_p = m.get("scale_mini_prev_bar_dir") or "-"
    mi = m.get("scale_mini_bar_dir") or m.get("scale_mini_dir") or "-"
    mili = m.get("scale_mili_dir") or "-"
    micro = micro_regime_token(m.get("scale_micro_regime"))
    return f"SCALE: tcn={tcn} tape={tape} adapted={adapted} micro={micro} mi_p={mi_p} mi={mi} mili={mili}"
