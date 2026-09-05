"""Sweep de horizonte N barras no TF SSOT (1HZ75V M5)."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.tf_sweep_symbols import resolve_sweep_symbols
from src.domain.config_knobs import load_settings_json


DEFAULT_N_BARS: tuple[int, ...] = (15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
DEFAULT_DURATION_MINUTES: tuple[int, ...] = DEFAULT_N_BARS
M1_SECONDS = 60
M5_SECONDS = 300
D1_SECONDS = 86400


def duration_minutes_for_n(n_bars: int, *, micro_seconds: int = M1_SECONDS) -> int:
    """Contrato em minutos = N velas vezes minutos da barra micro."""
    n = max(1, int(n_bars))
    micro = max(1, int(micro_seconds))
    return n * max(1, micro // 60)


def parse_n_bars(raw: Any) -> tuple[int, ...]:
    """Normaliza a grade de N; vazio cai no default M15..H1 de 5 em 5."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return DEFAULT_N_BARS
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        n = max(1, int(item))
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return tuple(out) if out else DEFAULT_N_BARS


def parse_duration_minutes(raw: Any) -> tuple[int, ...] | None:
    """Grade de duracoes em minutos; None se ausente/vazia."""
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    out: list[int] = []
    seen: set[int] = set()
    for item in raw:
        d = max(1, int(item))
        if d in seen:
            continue
        seen.add(d)
        out.append(d)
    return tuple(out) if out else None


def n_bars_from_durations(durations_m: tuple[int, ...] | list[int], *, micro_seconds: int) -> tuple[int, ...]:
    """Converte minutos de contrato em N velas; exige divisao exata."""
    micro_m = max(1, int(micro_seconds) // 60)
    out: list[int] = []
    for dur in durations_m:
        d = max(1, int(dur))
        if d % micro_m != 0:
            raise ValueError(f"duration_minutes={d} nao e multiplo da barra micro={micro_m}m")
        out.append(d // micro_m)
    return tuple(out)


def resolve_payout_for_sweep(full: dict[str, Any], block: dict[str, Any]) -> float:
    """Resolve payout de breakeven do bloco sweep ou risk_management.params."""
    if block.get("payout_for_breakeven") is not None:
        return float(block["payout_for_breakeven"])
    risk = full.get("risk_management") if isinstance(full.get("risk_management"), dict) else {}
    params = risk.get("params") if isinstance(risk, dict) and isinstance(risk.get("params"), dict) else {}
    if isinstance(params, dict) and params.get("payout_estimate") is not None:
        return float(params["payout_estimate"])
    return 0.72


def load_horizon_sweep_knobs(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Le deep_learning.horizon_sweep com defaults fail-closed."""
    full = settings if isinstance(settings, dict) else load_settings_json()
    dl = full.get("deep_learning") if isinstance(full.get("deep_learning"), dict) else {}
    data = full.get("data_handler") if isinstance(full.get("data_handler"), dict) else {}
    raw = dl.get("horizon_sweep") if isinstance(dl, dict) else None
    block = raw if isinstance(raw, dict) else {}
    payout = resolve_payout_for_sweep(full, block)
    micro = (
        int(data["micro_granularity"])
        if isinstance(data, dict) and data.get("micro_granularity") is not None
        else M1_SECONDS
    )
    raw_n = block.get("n_bars")
    if isinstance(raw_n, (list, tuple)) and raw_n:
        bars = list(parse_n_bars(raw_n))
    else:
        durations = parse_duration_minutes(block.get("duration_minutes"))
        if durations is not None:
            bars = list(n_bars_from_durations(durations, micro_seconds=micro))
        else:
            bars = list(parse_n_bars(None))
    ops_dur = block.get("ops_contract_duration_minutes")
    return {
        "enabled": bool(block.get("enabled", True)),
        "run_in_launch_train": bool(block.get("run_in_launch_train", True)),
        "auto_promote": bool(block.get("auto_promote", True)),
        "ops_contract_duration_minutes": max(1, int(5 if ops_dur is None else ops_dur)),
        "quiet_train_logs": bool(block.get("quiet_train_logs", True)),
        "train_deploy_retries": max(1, min(8, int(block.get("train_deploy_retries", 1)))),
        "disable_infra_during_sweep": bool(block.get("disable_infra_during_sweep", True)),
        "min_edge_vs_breakeven": float(block.get("min_edge_vs_breakeven", 0.03)),
        "min_settle_n": max(1, int(block.get("min_settle_n", 16))),
        "min_history_bars": max(0, int(block.get("min_history_bars", 800))),
        "payout_for_breakeven": payout,
        "weight_edge": float(block.get("weight_edge", 1.0)),
        "weight_brier": float(block.get("weight_brier", 0.5)),
        "weight_sharpness": float(block.get("weight_sharpness", 0.25)),
        "weight_meta_ir": float(block.get("weight_meta_ir", 0.25)),
        "leaderboard_path": str(block.get("leaderboard_path", "data/dl/sweep/leaderboard.json")),
        "artifact_root": str(block.get("artifact_root", "data/dl/sweep")),
        "soft_max_brier": float(block.get("soft_max_brier", 0.26)),
        "n_bars": bars,
        "duration_minutes": [duration_minutes_for_n(n, micro_seconds=micro) for n in bars],
        "symbols": resolve_sweep_symbols(block),
    }


def build_horizon_candidates(
    settings: dict[str, Any] | None = None,
    *,
    n_bars: tuple[int, ...] | list[int] | None = None,
) -> list[dict[str, Any]]:
    """Candidatos H{N} no relogio SSOT, sem reescalar lookback/history."""
    full = settings if isinstance(settings, dict) else {}
    data = full.get("data_handler") if isinstance(full.get("data_handler"), dict) else {}
    dl = full.get("deep_learning") if isinstance(full.get("deep_learning"), dict) else {}
    micro = (
        int(data["micro_granularity"])
        if isinstance(data, dict) and data.get("micro_granularity") is not None
        else M5_SECONDS
    )
    macro = int(data["granularity"]) if isinstance(data, dict) and data.get("granularity") is not None else D1_SECONDS
    mini = (
        int(data["mini_granularity"]) if isinstance(data, dict) and data.get("mini_granularity") is not None else micro
    )
    lookback = int(dl["lookback"]) if isinstance(dl, dict) and dl.get("lookback") is not None else 480
    history = (
        int(dl["training_history_bars"])
        if isinstance(dl, dict) and dl.get("training_history_bars") is not None
        else 5000
    )
    h_block = dl.get("horizon_sweep") if isinstance(dl, dict) and isinstance(dl.get("horizon_sweep"), dict) else {}
    if n_bars is not None:
        bars = parse_n_bars(n_bars)
    else:
        raw_n = h_block.get("n_bars")
        if isinstance(raw_n, (list, tuple)) and raw_n:
            bars = parse_n_bars(raw_n)
        else:
            durations = parse_duration_minutes(h_block.get("duration_minutes"))
            if durations is not None:
                bars = n_bars_from_durations(durations, micro_seconds=micro)
            else:
                bars = parse_n_bars(None)
    out: list[dict[str, Any]] = []
    for n in bars:
        out.append(
            {
                "tf": f"H{n}",
                "enabled": True,
                "micro_granularity": micro,
                "macro_granularity": macro,
                "mini_granularity": mini,
                "duration": duration_minutes_for_n(n, micro_seconds=micro),
                "duration_unit": "m",
                "lookback": lookback,
                "history_bars": history,
                "label_horizon_bars": n,
                "train_timeframe": "micro",
            }
        )
    return out
