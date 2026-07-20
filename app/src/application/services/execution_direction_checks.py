"""Checagens preliminares de direcao e gates sniper."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import (
    direction_margin_from_probability,
    passes_execution_quality,
)
from src.application.services.execution_quality_gate_meta import evaluate_meta_payoff_quality
from src.application.services.execution_quality_gate_microstructure import is_hard_quality_reject_reason
from src.application.services.execution_sniper_gates import (
    apply_bb_squeeze_requirement,
    apply_hurst_noise_veto,
)
from src.application.services.force_trade_mode import force_trade_every_cycle, synthesize_force_direction
from src.application.services.meta_payoff_regression import veto_calibration_neutral_drift
from src.domain.models.trade import TradeDirection
from src.domain.risk.soft_recovery_policy import negative_zscore_veto_floor_for_risk
from src.domain.risk.stake_sizing import metric_float


D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO = 0.55
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
    """True quando a calibracao bloqueou o sinal na zona neutra."""
    return (
        str(metrics.get("gate_reason") or "") == _NEUTRAL_CLAMP
        or str(metrics.get("calibration_mode") or "") == _NEUTRAL_CLAMP
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
            "exec_direction": dl_dir.name,
            "resolved_direction": dl_dir.name,
            "direction_inverted": False,
            "meta_direction_flip": False,
            "direction_call_score": call,
            "direction_put_score": put,
            "direction_margin": direction_margin_from_probability(prob, direction=dl_dir.name),
        }
    )
    return call if dl_dir == TradeDirection.CALL else put


def sync_entry_metrics(entry: dict, metrics: dict) -> None:
    """Propaga metricas resolvidas de volta ao entry do ciclo."""
    entry["metrics"].update(metrics) if isinstance(entry.get("metrics"), dict) else entry.setdefault("metrics", metrics)


def has_meta_zscore_telemetry(metrics: dict) -> bool:
    """True quando ha Z-Score meta com amostras suficientes."""
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        return False
    s = metrics.get("edge_zscore_samples")
    return s is None or int(s) >= 2


def meta_zscore_soft_ok(metrics: dict, *, risk_manager: Any | None = None) -> bool:
    """True quando o Z-Score meta esta acima do piso soft de recovery."""
    floor = negative_zscore_veto_floor_for_risk(risk_manager)
    return metric_float(metrics, "meta_payoff_edge_zscore", "edge_zscore", default=0.0) >= floor


def reject_on_quality_gate(
    entry: dict,
    metrics: dict,
    gate_probe: dict,
    exec_cfg_dict: dict,
    *,
    risk_manager: Any | None = None,
    recovery_active: bool = False,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> bool:
    """Rejeita por microestrutura ou margem de direcao insuficiente."""
    if force_trade_every_cycle(exec_cfg_dict):
        for k in ("quality_guard_reject", "regime_skip_cycle", "quality_gate_reason"):
            metrics.pop(k, None)
            gate_probe.pop(k, None)
        return False
    kw = {
        "exec_cfg": exec_cfg_dict,
        "risk_manager": risk_manager,
        "skipped_cycles_counter": skipped_cycles_counter,
        "orch": orch,
    }
    _ = (entry, recovery_active)
    if has_meta_zscore_telemetry(gate_probe):
        evaluate_meta_payoff_quality(gate_probe, **kw)
    passed = passes_execution_quality(gate_probe, **kw)
    for k in (
        "execution_gate_state",
        "quality_gate_regime",
        "direction_margin",
        "quality_min_direction_margin",
    ):
        if k in gate_probe:
            metrics[k] = gate_probe[k]
    reason = gate_probe.get("quality_gate_reason")
    if not passed and is_hard_quality_reject_reason(reason):
        metrics["quality_guard_reject"] = True
        metrics["regime_skip_cycle"] = True
        metrics["quality_gate_reason"] = reason
        sync_entry_metrics(entry, metrics)
        return True
    for k in ("quality_guard_reject", "regime_skip_cycle", "quality_gate_reason"):
        metrics.pop(k, None)
        gate_probe.pop(k, None)
    return False


def apply_technical_agreement(metrics: dict, dl_dir: TradeDirection, prob: float, exec_cfg: dict) -> tuple[float, bool]:
    """Ajusta probabilidade por consenso tecnico; veto de discordancia desativado."""
    call_votes, put_votes = int(metrics.get("call_votes", 0)), int(metrics.get("put_votes", 0))
    total = call_votes + put_votes
    if total <= 0:
        return prob, False
    opp = put_votes / total if dl_dir == TradeDirection.CALL else call_votes / total
    _ = exec_cfg
    adjusted = prob
    if (1.0 - opp) >= 0.80:
        adjusted = min(1.0, prob + 0.05) if dl_dir == TradeDirection.CALL else max(0.0, prob - 0.05)
    return adjusted, False


def sniper_cfg(exec_cfg_dict: dict, orch: Any | None) -> tuple[dict, dict]:
    """Extrai configs de squeeze e indicator_gating para os gates sniper."""
    squeeze = exec_cfg_dict.get("bb_width_adaptive_squeeze")
    squeeze_cfg = squeeze if isinstance(squeeze, dict) else {}
    gating_cfg: dict = {}
    if orch is not None and hasattr(orch, "config"):
        dl = orch.config.get("deep_learning", {}) if isinstance(orch.config, dict) else {}
        raw = dl.get("indicator_gating") if isinstance(dl, dict) else None
        if isinstance(raw, dict):
            gating_cfg = raw
    return squeeze_cfg, gating_cfg


def initial_direction_checks(
    entry: dict, exec_cfg_dict: dict, *, orch: Any | None = None
) -> tuple[TradeDirection, dict, float] | None:
    """Aplica clamps, sniper gates e discordance antes da resolucao final."""
    metrics = dict(entry.get("metrics") or {})
    force = force_trade_every_cycle(exec_cfg_dict)
    if _is_neutral_clamp(metrics):
        metrics["gate_reason"] = (
            None if str(metrics.get("gate_reason") or "") == _NEUTRAL_CLAMP else metrics.get("gate_reason")
        )
        metrics["calibration_mode"] = "calibrated"
        metrics.pop("quality_guard_reject", None)
        metrics.pop("regime_skip_cycle", None)
        sync_entry_metrics(entry, metrics)
    dl_dir = infer_dl_direction(entry)
    if force and dl_dir is None:
        dl_dir = synthesize_force_direction(entry)
    if is_technically_blocked(entry) or dl_dir is None or veto_calibration_neutral_drift(metrics):
        if metrics.get("quality_guard_reject") or metrics.get("gate_reason"):
            sync_entry_metrics(entry, metrics)
        return None
    squeeze_cfg, gating_cfg = sniper_cfg(exec_cfg_dict, orch)
    anomaly = squeeze_cfg.get("anomaly_ratio", D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO)
    metrics["bb_width_anomaly_ratio"] = float(anomaly)
    _ = apply_hurst_noise_veto(metrics, gating_cfg) or apply_bb_squeeze_requirement(metrics, squeeze_cfg)
    prob = direction_prob(entry)
    if prob is None:
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    prob, should_veto = apply_technical_agreement(metrics, dl_dir, clamp01(prob), exec_cfg_dict)
    _ = should_veto
    return dl_dir, metrics, prob
