"""Checagens preliminares de direcao e gates sniper."""

from __future__ import annotations

from typing import Any

from src.application.services.deep_learning.dl_indicator_config import load_bb_width_anomaly_ratio
from src.application.services.execution_adverse_path_gate import apply_adverse_micro_path_gate
from src.application.services.execution_direction_discordance import (
    _macro_indicator_float,
    apply_technical_agreement,
    resolve_formed_candle_direction,
)
from src.application.services.execution_price_zone_gate import apply_price_zone_gate_with_starvation
from src.application.services.execution_quality_gate_microstructure import resolve_skipped_cycles
from src.application.services.execution_quality_gate_starvation import starvation_decay_factor
from src.application.services.execution_sniper_gates import hurst_regime_allowed
from src.application.services.force_trade_mode import force_trade_every_cycle, synthesize_force_direction
from src.domain.models.trade import TradeDirection
from src.domain.risk.soft_recovery_policy import negative_zscore_veto_floor_for_risk
from src.domain.risk.stake_sizing import metric_float


_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})
_NEUTRAL_PIVOT_EPS = 1e-9
_NEUTRAL_CLAMP = "neutral_clamp"


def direction_prob(entry: dict) -> float | None:
    """Le probabilidade calibrada ou bruta do candidato."""
    m = entry.get("metrics") or {}
    cal = m.get("calibrated_prob")
    if cal is not None:
        return float(cal)
    raw = m.get("raw_prob")
    return float(raw) if raw is not None else None


def direction_pivot(metrics: dict) -> float:
    """Calcula pivô neutro a partir dos limiares dinamicos CALL/PUT."""
    c, p = metrics.get("dynamic_call_threshold"), metrics.get("dynamic_put_threshold")
    return (float(c) + float(p)) * 0.5 if c is not None and p is not None else 0.5


def _is_neutral_clamp(metrics: dict) -> bool:
    """True quando a calibracao, gate_reason ou sinal indica zona neutra sem alinhamento de vela."""
    if isinstance(metrics, dict) and metrics.get("candle_color_direction") is not None:
        return False
    reason = str(metrics.get("gate_reason") or "")
    cal_mode = str(metrics.get("calibration_mode") or "")
    status = str(metrics.get("signal_status") or "")
    action = str(metrics.get("action") or "")
    cal_m = metric_float(metrics, "cal_margin", "calibrated_margin", "cal_m", "direction_margin", default=0.0)
    raw_m = metric_float(metrics, "raw_margin", "raw_m", default=0.0)
    floor = metric_float(
        metrics, "quality_min_direction_margin", "min_direction_margin", "neutral_floor", "floor", default=0.0
    )
    if floor > 0.0:
        margin = cal_m if cal_m > 0.0 else raw_m
        if margin > 0.0 and margin < floor:
            return True
    return (
        reason in {_NEUTRAL_CLAMP, "neutral", "NEUTRAL", "NO_EDGE_NEUTRAL", "NEUTRO", "NEUTRO_SKIP"}
        or cal_mode in {_NEUTRAL_CLAMP, "neutral", "NEUTRAL"}
        or status in {"NEUTRAL", "neutral", "NEUTRO", "NEUTRO_SKIP"}
        or action in {"NEUTRAL", "neutral"}
    )


def infer_dl_direction(entry: dict) -> TradeDirection | None:
    """Infere CALL/PUT a partir da probabilidade, ou None se indefinido."""
    metrics = entry.get("metrics") or {}
    d = entry.get("direction")
    if d is not None:
        return d
    p = direction_prob(entry)
    if p is None:
        return None
    pivot = direction_pivot(metrics)
    return TradeDirection.CALL if float(p) + _NEUTRAL_PIVOT_EPS >= float(pivot) else TradeDirection.PUT


def is_technically_blocked(entry: dict) -> bool:
    """True quando deploy falhou ou o gate_reason e bloqueio tecnico."""
    m = entry.get("metrics") or {}
    return m.get("deploy_ok") is False or str(m.get("gate_reason") or "") in _TECHNICAL_BLOCKS


def clamp01(v: float) -> float:
    """Restringe valor ao intervalo unitario [0, 1]."""
    return max(0.0, min(1.0, float(v)))


def seed_direction_metrics(metrics: dict, *, dl_dir: TradeDirection, prob: float) -> float:
    """Inicializa scores e margem direcional a partir da probabilidade TCN."""
    call, put = prob, 1.0 - prob
    metrics.update(
        {
            "dl_direction": dl_dir.name,
            "resolved_direction": dl_dir.name,
            "raw_prob": prob,
            "raw_call_prob": call,
            "raw_put_prob": put,
            "raw_margin": abs(prob - 0.5),
            "calibrated_prob": prob,
            "calibrated_call_prob": call,
            "calibrated_put_prob": put,
            "direction_margin": abs(prob - 0.5),
        }
    )
    return max(call, put)


def sync_entry_metrics(entry: dict, metrics: dict) -> None:
    """Propaga metricas resolvidas de volta ao entry do ciclo."""
    entry["metrics"].update(metrics) if isinstance(entry.get("metrics"), dict) else entry.setdefault("metrics", metrics)


def meta_zscore_soft_ok(metrics: dict, *, risk_manager: Any | None = None) -> bool:
    """True quando o Z-Score meta esta acima do piso soft de recovery."""
    floor = negative_zscore_veto_floor_for_risk(risk_manager)
    return metric_float(metrics, "meta_payoff_edge_zscore", "edge_zscore", default=0.0) >= floor


def sniper_cfg(exec_cfg_dict: dict, orch: Any | None) -> tuple[dict, dict]:
    """Extrai configs de squeeze e indicator_gating para os gates sniper."""
    squeeze = exec_cfg_dict.get("bb_width_adaptive_squeeze")
    squeeze_cfg = dict(squeeze) if isinstance(squeeze, dict) else {}
    if "anomaly_ratio" not in squeeze_cfg:
        squeeze_cfg["anomaly_ratio"] = load_bb_width_anomaly_ratio()
    gating_cfg: dict = {}
    if orch is not None and hasattr(orch, "config") and isinstance(orch.config, dict):
        dl = orch.config.get("deep_learning", {})
        raw = dl.get("indicator_gating") if isinstance(dl, dict) else None
        if isinstance(raw, dict):
            gating_cfg = raw
    return squeeze_cfg, gating_cfg


def initial_direction_checks(
    entry: dict,
    exec_cfg_dict: dict,
    *,
    orch: Any | None = None,
    skipped_cycles_counter: int | None = None,
) -> tuple[TradeDirection, dict, float] | None:
    """Aplica clamps, sniper gates e discordance antes da resolucao final."""
    metrics = dict(entry.get("metrics") or {})
    for sticky in (
        "side_eq_gate_done",
        "side_eq_blocked",
        "side_eq_flipped",
        "side_eq_flip_from",
        "side_eq_margin_boost",
        "execution_candidate_ready",
    ):
        metrics.pop(sticky, None)
    force = force_trade_every_cycle(exec_cfg_dict)
    dl_dir = infer_dl_direction(entry)
    if dl_dir is not None:
        resolve_formed_candle_direction(metrics, dl_dir)
    if _is_neutral_clamp(metrics):
        if not force:
            metrics["gate_reason"] = metrics.get("gate_reason") or _NEUTRAL_CLAMP
            metrics["calibration_mode"] = metrics.get("calibration_mode") or _NEUTRAL_CLAMP
            metrics["quality_guard_reject"] = True
            metrics["regime_skip_cycle"] = True
            metrics["signal_status"] = "SKIP"
            metrics["execute"] = False
            entry["execute"] = False
            sync_entry_metrics(entry, metrics)
            return None
        metrics["gate_reason"] = (
            None if str(metrics.get("gate_reason") or "") == _NEUTRAL_CLAMP else metrics.get("gate_reason")
        )
        metrics["calibration_mode"] = "calibrated"
        metrics.pop("quality_guard_reject", None)
        metrics.pop("regime_skip_cycle", None)
        metrics.pop("signal_status", None)
        sync_entry_metrics(entry, metrics)
    dl_dir = infer_dl_direction(entry)
    if force and dl_dir is None:
        dl_dir = synthesize_force_direction(entry)
    if is_technically_blocked(entry) or dl_dir is None:
        if metrics.get("quality_guard_reject") or metrics.get("gate_reason"):
            sync_entry_metrics(entry, metrics)
        return None
    skipped = resolve_skipped_cycles(skipped_cycles_counter=skipped_cycles_counter, orch=orch)
    adverse = apply_adverse_micro_path_gate(metrics, dl_dir, exec_cfg_dict, skipped_cycles_counter=skipped, orch=orch)
    if not force and adverse:
        sync_entry_metrics(entry, metrics)
        return None
    squeeze_cfg, gating_cfg = sniper_cfg(exec_cfg_dict, orch)
    metrics["bb_width_anomaly_ratio"] = float(squeeze_cfg["anomaly_ratio"])
    decay = starvation_decay_factor(skipped, exec_cfg=exec_cfg_dict)
    adx_min = float(gating_cfg.get("adx_min", 0.0)) * float(decay)
    metrics["quality_min_adx"] = float(adx_min)
    metrics["quality_adx_decay_factor"] = float(decay)
    if adx_min > 0.0 and not force:
        adx = _macro_indicator_float(metrics, "adx")
        if adx is not None:
            metrics["quality_adx"] = float(adx)
            if float(adx) + 1e-12 < adx_min:
                metrics["quality_guard_reject"] = True
                metrics["regime_skip_cycle"] = True
                metrics["gate_reason"] = "adx_min"
                metrics["quality_adx_detail"] = f"adx={float(adx):.3f} < min={float(adx_min):.3f}"
                sync_entry_metrics(entry, metrics)
                return None
    if not force and bool(gating_cfg.get("enabled", False)):
        hurst = _macro_indicator_float(metrics, "hurst")
        if not hurst_regime_allowed(hurst, gating_cfg):
            metrics["quality_guard_reject"] = True
            metrics["regime_skip_cycle"] = True
            metrics["gate_reason"] = "hurst_missing" if hurst is None else "hurst_noise"
            sync_entry_metrics(entry, metrics)
            return None
    prob = direction_prob(entry)
    if prob is None:
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    prob, should_veto = apply_technical_agreement(
        metrics, dl_dir, clamp01(prob), exec_cfg_dict, skipped_cycles_counter=skipped, orch=orch
    )
    if should_veto and not force:
        metrics["quality_guard_reject"] = True
        metrics["regime_skip_cycle"] = True
        metrics["gate_reason"] = str(metrics.get("gate_reason") or "indicator_discordance")
        sync_entry_metrics(entry, metrics)
        return None
    zone_reason = apply_price_zone_gate_with_starvation(
        metrics,
        dl_dir,
        exec_cfg_dict,
        tcn_direction=dl_dir,
        skipped_cycles_counter=skipped,
        orch=orch,
        force=force,
    )
    if zone_reason is not None:
        metrics["quality_guard_reject"] = True
        metrics["regime_skip_cycle"] = True
        metrics["gate_reason"] = zone_reason
        sync_entry_metrics(entry, metrics)
        return None
    metrics["dl_direction"] = dl_dir.name
    metrics["resolved_direction"] = dl_dir.name
    sync_entry_metrics(entry, metrics)
    return dl_dir, metrics, prob
