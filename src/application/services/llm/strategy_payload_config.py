"""Limares e rotulos do payload Sniper lidos do JSON (strategy / strategy_thresholds)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyPayloadConfig:
    """Parametros de traducao quant -> tokens e ordem da linha sniper."""

    hurst_persist: str
    hurst_anti: str
    hurst_random: str
    hurst_na: str
    zscore_high: str
    zscore_low: str
    zscore_neutral: str
    zscore_na: str
    entropy_low: str
    entropy_high: str
    entropy_extreme: str
    entropy_na: str
    velocity_pos: str
    velocity_neg: str
    velocity_na: str
    acceleration_up: str
    acceleration_down: str
    acceleration_flat: str
    acceleration_na: str
    payload_token_order: tuple[str, ...]
    field_labels: dict[str, str]
    mtf_token_key: str
    sym_token_key: str
    pair_separator: str
    kv_separator: str


DEFAULT_STRATEGY_PAYLOAD_CONFIG = StrategyPayloadConfig(
    hurst_persist="persist",
    hurst_anti="anti",
    hurst_random="random",
    hurst_na="na",
    zscore_high="high",
    zscore_low="low",
    zscore_neutral="neutral",
    zscore_na="na",
    entropy_low="low",
    entropy_high="high",
    entropy_extreme="extreme",
    entropy_na="na",
    velocity_pos="pos",
    velocity_neg="neg",
    velocity_na="na",
    acceleration_up="accel_up",
    acceleration_down="accel_down",
    acceleration_flat="accel_flat",
    acceleration_na="na",
    payload_token_order=("hurst", "zscore", "entropy", "velocity", "acceleration"),
    field_labels={"hurst": "h", "zscore": "z", "entropy": "e", "velocity": "v", "acceleration": "a"},
    mtf_token_key="MTF",
    sym_token_key="SYM",
    pair_separator=", ",
    kv_separator="=",
)


def resolve_strategy_payload_config(root: dict[str, Any]) -> StrategyPayloadConfig:
    """Constroi config a partir de ``strategy`` ou ``strategy_thresholds`` no JSON."""
    st = root.get("strategy")
    if not isinstance(st, dict):
        legacy = root.get("strategy_thresholds")
        st = legacy if isinstance(legacy, dict) else {}
    th = st.get("thresholds") if isinstance(st.get("thresholds"), dict) else {}
    pl = st.get("payload") if isinstance(st.get("payload"), dict) else {}

    hst = th.get("hurst") if isinstance(th.get("hurst"), dict) else {}
    zsc = th.get("zscore") if isinstance(th.get("zscore"), dict) else {}
    ent = th.get("entropy") if isinstance(th.get("entropy"), dict) else {}
    vel = th.get("velocity") if isinstance(th.get("velocity"), dict) else {}
    acc = th.get("acceleration") if isinstance(th.get("acceleration"), dict) else {}

    order_raw = pl.get("token_order") or ("hurst", "zscore", "entropy", "velocity", "acceleration")
    order = (
        tuple(str(x).strip() for x in order_raw if str(x).strip())
        if isinstance(order_raw, list)
        else DEFAULT_STRATEGY_PAYLOAD_CONFIG.payload_token_order
    )

    fl_raw = pl.get("field_labels") if isinstance(pl.get("field_labels"), dict) else {}
    field_labels = {
        "hurst": str(fl_raw.get("hurst", "h")).strip() or "h",
        "zscore": str(fl_raw.get("zscore", "z")).strip() or "z",
        "entropy": str(fl_raw.get("entropy", "e")).strip() or "e",
        "velocity": str(fl_raw.get("velocity", "v")).strip() or "v",
        "acceleration": str(fl_raw.get("acceleration", "a")).strip() or "a",
    }

    base = DEFAULT_STRATEGY_PAYLOAD_CONFIG
    return StrategyPayloadConfig(
        hurst_persist=str(hst.get("persist", base.hurst_persist)).strip() or base.hurst_persist,
        hurst_anti=str(hst.get("anti", base.hurst_anti)).strip() or base.hurst_anti,
        hurst_random=str(hst.get("random", base.hurst_random)).strip() or base.hurst_random,
        hurst_na=str(hst.get("na", base.hurst_na)).strip() or base.hurst_na,
        zscore_high=str(zsc.get("high", base.zscore_high)).strip() or base.zscore_high,
        zscore_low=str(zsc.get("low", base.zscore_low)).strip() or base.zscore_low,
        zscore_neutral=str(zsc.get("neutral", base.zscore_neutral)).strip() or base.zscore_neutral,
        zscore_na=str(zsc.get("na", base.zscore_na)).strip() or base.zscore_na,
        entropy_low=str(ent.get("low", base.entropy_low)).strip() or base.entropy_low,
        entropy_high=str(ent.get("high", base.entropy_high)).strip() or base.entropy_high,
        entropy_extreme=str(ent.get("extreme", base.entropy_extreme)).strip() or base.entropy_extreme,
        entropy_na=str(ent.get("na", base.entropy_na)).strip() or base.entropy_na,
        velocity_pos=str(vel.get("pos", base.velocity_pos)).strip() or base.velocity_pos,
        velocity_neg=str(vel.get("neg", base.velocity_neg)).strip() or base.velocity_neg,
        velocity_na=str(vel.get("na", base.velocity_na)).strip() or base.velocity_na,
        acceleration_up=str(acc.get("up", base.acceleration_up)).strip() or base.acceleration_up,
        acceleration_down=str(acc.get("down", base.acceleration_down)).strip() or base.acceleration_down,
        acceleration_flat=str(acc.get("flat", base.acceleration_flat)).strip() or base.acceleration_flat,
        acceleration_na=str(acc.get("na", base.acceleration_na)).strip() or base.acceleration_na,
        payload_token_order=order if order else base.payload_token_order,
        field_labels=field_labels,
        mtf_token_key=str(pl.get("mtf_key", base.mtf_token_key)).strip() or base.mtf_token_key,
        sym_token_key=str(pl.get("sym_key", base.sym_token_key)).strip() or base.sym_token_key,
        pair_separator=str(pl.get("pair_separator", base.pair_separator)),
        kv_separator=str(pl.get("kv_separator", base.kv_separator)),
    )
