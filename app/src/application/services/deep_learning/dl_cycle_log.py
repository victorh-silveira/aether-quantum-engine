"""Formatacao compacta de logs do ciclo Deep Learning."""

from src.application.services.deep_learning.dl_cycle_brief import (
    _format_bias_token,
    build_dl_cycle_brief,
    build_dl_cycle_brief_key,
)
from src.application.services.deep_learning.dl_gating import resolve_edge
from src.application.services.execution_direction_resolver import infer_dl_direction, is_technically_blocked
from src.application.services.execution_scale_vision import format_scale_audit_line
from src.application.services.log_dedupe import log_info_if_changed
from src.application.services.market_audit_candle import log_closed_candle_outcomes
from src.application.services.market_audit_log import format_cluster_audit_line, resolve_cluster_timeframe
from src.domain.risk.stake_sizing import metric_float, raw_side_from_metrics


def _best_directional_signal(decisions: dict[str, dict]) -> tuple[str, float] | None:
    """Retorna simbolo e sinal efetivo mais forte entre candidatos com direcao DL."""
    best_symbol = None
    best_score = -1.0
    for symbol, entry in decisions.items():
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            continue
        if infer_dl_direction(entry) is None:
            continue
        score = metric_float(metrics, "trade_score", "conviction", default=0.0)
        raw_side = raw_side_from_metrics(metrics)
        effective = max(score, raw_side)
        if effective > best_score:
            best_score = effective
            best_symbol = symbol
    if best_symbol is None:
        return None
    return best_symbol, best_score


def build_dl_cycle_summary(
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
    pending_loss_total: float,
) -> str:
    """Monta uma unica linha resumo com candidatos e bloqueios do ciclo DL."""
    exec_tokens: list[str] = []
    bias_tokens: list[str] = []
    skip_tokens: list[str] = []
    train_tokens: list[str] = []
    for symbol, entry in decisions.items():
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            train_tokens.append(symbol)
            continue
        if is_technically_blocked(entry):
            gate = metrics.get("gate_reason") or "block"
            skip_tokens.append(f"{symbol}:{gate}")
            continue
        if infer_dl_direction(entry) is None:
            skip_tokens.append(f"{symbol}:sem_dir")
            continue
        direction = entry.get("direction")
        conv = metric_float(metrics, "trade_score", "conviction", default=0.0)
        val_acc = float(metrics.get("val_accuracy", 0.0))
        raw = metrics.get("raw_prob")
        gap_s = ""
        if raw is not None:
            gap_s = f":e{resolve_edge(float(raw)):.2f}"
        label = f"{symbol}:{direction.name if direction else metrics.get('exec_direction', '?')}:{conv:.2f}:v{val_acc:.2f}{gap_s}"
        if metrics.get("execute", True):
            suffix = ""
            if metrics.get("bypass_val_acc"):
                suffix = ":bypass"
            elif recovery_active:
                suffix = ":rec"
            exec_tokens.append(f"{label}{suffix}")
        bias = _format_bias_token(symbol, entry)
        if bias:
            bias_tokens.append(bias)
    mode = f"RECOVERY pend=${pending_loss_total:.0f}" if recovery_active else "NORMAL"
    exec_part = ",".join(exec_tokens) if exec_tokens else "none"
    bias_part = ",".join(bias_tokens[:5]) if bias_tokens else "-"
    skip_part = ",".join(skip_tokens[:5]) if skip_tokens else "-"
    extra = f" +{len(skip_tokens) - 5}" if len(skip_tokens) > 5 else ""
    train_part = f" | treino=[{','.join(train_tokens)}]" if train_tokens else ""
    return f"DL | {mode} | exec=[{exec_part}] | bias=[{bias_part}] | skip=[{skip_part}{extra}]{train_part}"


def log_dl_cycle_summary(
    logger,
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
    pending_loss_total: float,
    orch=None,
) -> None:
    """Registra resumo detalhado em DEBUG e linha curta em INFO sem repetir conteudo."""
    logger.debug(
        build_dl_cycle_summary(
            decisions,
            recovery_active=recovery_active,
            pending_loss_total=pending_loss_total,
        )
    )
    brief = build_dl_cycle_brief(
        decisions,
        recovery_active=recovery_active,
    )
    timeframe = resolve_cluster_timeframe(getattr(orch, "config", None) if orch is not None else None)
    cluster_line = format_cluster_audit_line(decisions, timeframe=timeframe)
    if orch is None:
        logger.info(cluster_line)
        return
    key_brief = build_dl_cycle_brief_key(
        decisions,
        recovery_active=recovery_active,
    )
    cycle_id = int(getattr(orch, "_active_cycle_id", 0) or 0)
    log_info_if_changed(orch, logger, f"dl_brief:{cycle_id}", key_brief, "%s", cluster_line)
    log_closed_candle_outcomes(logger, orch, decisions)
    _log_scale_lines(logger, decisions, orch=orch, cycle_id=cycle_id)
    _ = brief


def _log_scale_lines(logger, decisions: dict[str, dict], *, orch=None, cycle_id: int = 0) -> None:
    """Emite linha SCALE por simbolo com MACRO/MICRO/MINI/MILI."""
    for symbol, entry in decisions.items():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        line = metrics.get("scale_audit")
        if not line:
            if "scale_micro_dir" not in metrics and "scale_reason" not in metrics:
                continue
            line = format_scale_audit_line(metrics)
        if "MACRO=" not in str(line):
            continue
        message = f"[c{int(cycle_id)}|{symbol}] {line}"
        if orch is None:
            logger.debug("%s", message)
            continue
        logger.debug("%s", message)
        _ = (cycle_id, symbol, line)
