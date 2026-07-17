"""Formatos unificados de auditoria de mercado para terminal e monitoramento."""

from __future__ import annotations

from typing import Any


__all__ = [
    "format_cluster_audit_line",
    "format_execution_ticket_line",
    "format_indicators_audit_line",
    "format_settlement_audit_line",
    "pop_contract_audit",
    "resolve_cluster_timeframe",
    "resolve_predicted_edge",
    "resolve_settlement_tag",
    "resolve_stake_audit_context",
    "resolve_stake_mode_tag",
    "store_contract_audit",
]


def resolve_predicted_edge(metrics: dict[str, Any]) -> float:
    """Extrai edge continuo do meta-regressor a partir das metricas do ciclo."""
    raw = metrics.get("predicted_payoff_edge", metrics.get("meta_calibrated_payoff_score", 0.0))
    return float(raw or 0.0)


def resolve_cluster_timeframe(config: dict[str, Any] | None) -> str:
    """Resolve rotulo de timeframe do cluster a partir da granularidade configurada."""
    if not isinstance(config, dict):
        return "M5"
    data = config.get("data_handler")
    if not isinstance(data, dict):
        data = {}
    seconds = int(data.get("micro_granularity", data.get("granularity", 300)) or 300)
    if seconds >= 900:
        return "M15"
    if seconds >= 300:
        return "M5"
    if seconds >= 60:
        return f"M{max(1, seconds // 60)}"
    return f"S{seconds}"


def _indicator_snapshot(metrics: dict[str, Any]) -> dict[str, float]:
    """Consolida indicadores macro e micro em um unico mapa numerico."""
    merged: dict[str, float] = {}
    for bucket in ("indicators", "macro_indicators", "micro_indicators"):
        chunk = metrics.get(bucket)
        if not isinstance(chunk, dict):
            continue
        for key, raw in chunk.items():
            if raw is None:
                continue
            try:
                merged[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
    return merged


def _metric_float(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Le o primeiro campo numerico disponivel nas metricas."""
    for key in keys:
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def _veto_token(metrics: dict[str, Any]) -> str | None:
    """Resolve rotulo curto de veto de microestrutura ou quality gate."""
    for key in ("quality_gate_reason", "gate_reason"):
        raw = metrics.get(key)
        if not raw:
            continue
        token = str(raw).strip().upper().replace("-", "_")
        if token in {"NEUTRAL_CLAMP", "NEUTRO_CLAMP"}:
            return "NEUTRO_VETO"
        if token:
            return token
    if metrics.get("quality_guard_reject") or metrics.get("regime_skip_cycle"):
        return "NEUTRO_VETO"
    if metrics.get("execute") is False and metrics.get("deploy_ok") is not False:
        return "NEUTRO_VETO"
    return None


def _cluster_symbol_token(symbol: str, entry: dict[str, Any]) -> str:
    """Formata token de simbolo no resumo CLUSTER."""
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    direction = entry.get("direction")
    if direction is not None and hasattr(direction, "name"):
        side = str(direction.name)
    else:
        side = str(
            metrics.get("exec_direction") or metrics.get("dl_direction") or metrics.get("resolved_direction") or "FLAT"
        ).upper()
    veto = _veto_token(metrics)
    if veto is not None:
        return f"{symbol}: {side} ({veto})"
    raw_prob = _metric_float(metrics, "raw_prob", default=0.5)
    cal_prob = _metric_float(metrics, "calibrated_prob", "raw_prob", default=raw_prob)
    edge = resolve_predicted_edge(metrics)
    return f"{symbol}: {side} (Prob: {raw_prob:0.3f} Cal: {cal_prob:0.3f} Edge: {edge:+0.3f})"


def format_cluster_audit_line(
    decisions: dict[str, Any],
    *,
    timeframe: str = "M5",
) -> str:
    """Monta linha compacta de decisao do cluster por simbolo."""
    if not isinstance(decisions, dict) or not decisions:
        return f"[CLUSTER] {timeframe} || EMPTY"
    tokens = [
        _cluster_symbol_token(str(symbol), entry if isinstance(entry, dict) else {})
        for symbol, entry in decisions.items()
    ]
    return f"[CLUSTER] {timeframe} || " + " || ".join(tokens)


def format_indicators_audit_line(cycle_id: int, symbol: str, metrics: dict[str, Any]) -> str:
    """Monta linha compacta de indicadores e microestrutura alinhada por colunas."""
    _ = symbol
    snap = _indicator_snapshot(metrics)
    rsi = snap.get("rsi", 0.0)
    adx = snap.get("adx", 0.0)
    hurst = snap.get("hurst", 0.0)
    atr = snap.get("atr_norm", 0.0)
    bbw = snap.get("bb_width", 0.0)
    vol_r = snap.get("vol_ratio", snap.get("vol_ratio_short_long", 0.0))
    z_edge = _metric_float(metrics, "edge_zscore", "meta_payoff_edge_zscore", default=0.0)
    acc = _metric_float(metrics, "val_accuracy", default=0.0)
    return (
        f"[C{int(cycle_id):04d}] IND || "
        f"RSI: {rsi:>7.4f} | ADX: {adx:>7.4f} | HURST: {hurst:>7.4f} || "
        f"ATR: {atr:>8.4f} | BBW: {bbw:>8.4f} | VOL_R: {vol_r:>7.4f} || "
        f"Z: {z_edge:>+6.2f} | ACC: {acc:>6.4f}"
    )


def resolve_stake_mode_tag(mode_tag: str, linear_losses: int) -> str:
    """Compacta modo de sizing em rotulo curto (KELLY / DAL_Ln)."""
    tag = str(mode_tag or "KELLY").upper()
    if "ALEMBERT" in tag or tag.startswith("DAL"):
        return f"DAL_L{max(0, int(linear_losses))}"
    return "KELLY"


def resolve_stake_audit_context(rm: Any, *, balance_fallback: float | None = None) -> tuple[str, float, float]:
    """Le modo, pendencia e banca do ultimo sizing para a linha EXEC."""
    audit = getattr(rm, "_last_stake_audit", None)
    mode_tag = "KELLY"
    pending = float(rm.pending_loss_total()) if callable(getattr(rm, "pending_loss_total", None)) else 0.0
    bankroll = float(getattr(rm, "bankroll", getattr(rm, "initial_bankroll", 0.0)) or 0.0)
    if isinstance(audit, dict):
        return (
            str(audit.get("mode_tag") or mode_tag),
            float(audit.get("pending", pending)),
            float(audit.get("bankroll", bankroll)),
        )
    if isinstance(balance_fallback, (int, float)):
        bankroll = float(balance_fallback)
    return mode_tag, pending, bankroll


def format_execution_ticket_line(
    cycle_id: int,
    *,
    direction: str,
    symbol: str,
    stake: float,
    mode_tag: str,
    pending: float,
    bankroll: float,
    contract_id: int,
    payout: float,
) -> str:
    """Monta linha unica de risco e boleta EXEC."""
    return (
        f"[C{int(cycle_id):04d}] EXEC || {direction} [{symbol}] || "
        f"STAKE: {float(stake):.2f} ({mode_tag}) | "
        f"PEND: {float(pending):.2f} | BANCA: {float(bankroll):.2f} || "
        f"CID: {int(contract_id)} | PAY: {float(payout):.2f}"
    )


def resolve_settlement_tag(*, profit: float, linear_before: int) -> str:
    """Resolve sufixo de liquidacao (RESET_LINEAR / COOLDOWN_Ln)."""
    if float(profit) >= 0.0:
        return "RESET_LINEAR" if int(linear_before) > 0 else "FLAT_KEEP"
    return f"COOLDOWN_L{max(1, int(linear_before) + 1)}"


def format_settlement_audit_line(
    cycle_id: int,
    outcome: str,
    profit: float,
    direction: str,
    symbol: str,
    edge: float,
    *,
    settlement_tag: str | None = None,
) -> str:
    """Monta linha padronizada de liquidacao RESOLVED."""
    _ = (direction, symbol, edge)
    tag = settlement_tag or resolve_settlement_tag(profit=profit, linear_before=0)
    return f"[C{int(cycle_id):04d}] RESOLVED || STATUS: {str(outcome):<4} | P&L: {float(profit):>+7.2f} | {tag}"


def store_contract_audit(
    orch: Any,
    contract_id: int,
    *,
    symbol: str,
    direction: str,
    edge: float,
) -> None:
    """Persiste metadados de auditoria por contrato ate a liquidacao."""
    bag = getattr(orch, "_contract_audit", None)
    if bag is None:
        orch._contract_audit = {}
        bag = orch._contract_audit
    bag[int(contract_id)] = {
        "symbol": str(symbol),
        "direction": str(direction),
        "edge": float(edge),
    }


def pop_contract_audit(
    orch: Any,
    contract_id: int,
    *,
    contract: Any = None,
    symbol: str = "UNK",
) -> tuple[str, str, float]:
    """Recupera e remove metadados de auditoria de um contrato liquidado."""
    bag = getattr(orch, "_contract_audit", None) or {}
    snap = bag.pop(int(contract_id), None)
    if isinstance(snap, dict):
        return str(snap.get("symbol", symbol)), str(snap.get("direction", "UNK")), float(snap.get("edge", 0.0))
    dir_name = "UNK"
    if contract is not None:
        loss_dir = getattr(contract, "direction", None)
        if loss_dir is not None:
            dir_name = loss_dir.name
    return str(symbol), dir_name, 0.0
