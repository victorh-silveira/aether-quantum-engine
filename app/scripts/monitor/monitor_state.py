"""Estado compartilhado e resolucao de metas financeiras da sessao ativa."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


RANGE_BREAK_SYMBOLS = ("R_10", "R_25", "R_50", "R_75", "R_100")


@dataclass
class DashboardState:
    balance: float = 0.0
    session_start_balance: float = 0.0
    session_target_win: float = 0.0
    session_profit: float = 0.0
    compounding_rate: float = 0.01
    compounding_enabled: bool = True
    active_contracts: dict = field(default_factory=dict)
    last_telemetry: dict = field(default_factory=dict)
    trading_mode: str = "N/A"
    dl_arch: str = "tcn"
    decision_mode: str = "SELETIVO"
    active_symbols: tuple[str, ...] = RANGE_BREAK_SYMBOLS
    redis_url: str = ""
    redis_key_prefix: str = "aether"


@dataclass(frozen=True)
class SessionFinancials:
    start_balance: float
    target_win: float
    target_balance: float
    remaining: float
    progress_pct: float
    roi_pct: float
    goal_label: str


def _round_currency(value: float) -> float:
    return math.floor(max(0.0, float(value)) * 100) / 100


def resolve_session_financials(state: DashboardState) -> SessionFinancials:
    start = float(state.session_start_balance)
    target = float(state.session_target_win)
    if start <= 0.0 and state.balance > 0.0:
        start = float(state.balance)
    if target <= 0.0 and start > 0.0:
        rate = float(state.compounding_rate) if state.compounding_enabled else 0.0
        target = _round_currency(start * rate) if rate > 0.0 else 0.0
    profit = float(state.session_profit)
    target_balance = start + target if start > 0.0 else 0.0
    remaining = max(0.0, target - profit) if target > 0.0 else 0.0
    progress_pct = min(100.0, max(0.0, (profit / target * 100.0) if target > 0.0 else 0.0))
    roi_pct = (profit / start * 100.0) if start > 0.0 else 0.0
    rate_pct = float(state.compounding_rate) * 100.0
    goal_label = f"({rate_pct:.1f}% SES. ATIVA)"
    return SessionFinancials(
        start_balance=start,
        target_win=target,
        target_balance=target_balance,
        remaining=remaining,
        progress_pct=progress_pct,
        roi_pct=roi_pct,
        goal_label=goal_label,
    )


def decision_engine_label(state: DashboardState) -> str:
    arch = str(state.dl_arch or "tcn").upper()
    mode = str(state.decision_mode or "SELETIVO").upper()
    return f"DEEP_LEARNING {mode} ({arch} Engine)"


def active_symbols_label(state: DashboardState) -> str:
    symbols = state.active_symbols or RANGE_BREAK_SYMBOLS
    return "Símbolos ativos: " + ", ".join(symbols)
