"""Alvo de lucro por sessao ativa com juros compostos (sem stop loss)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


REDIS_SESSION_START_BALANCE_KEY = "session:current:start_balance"
REDIS_SESSION_TARGET_WIN_KEY = "session:current:target_win"


@dataclass(frozen=True)
class SessionTargets:
    """Metas financeiras calculadas para a sessao ativa corrente."""

    session_start_balance: float
    target_win: float
    compounding_rate: float


def _round_currency(value: float) -> float:
    """Arredonda valor monetario para baixo em centavos."""
    return math.floor(max(0.0, float(value)) * 100) / 100


class StopWinManager:
    """Calcula meta de stop win da sessao (4,31% composto, ou fixo em micro-banca)."""

    def __init__(self, risk_management: dict[str, Any] | None):
        self.risk_management = risk_management if isinstance(risk_management, dict) else {}
        raw_params = self.risk_management.get("params")
        self.params = raw_params if isinstance(raw_params, dict) else {}

    def is_compounding_enabled(self) -> bool:
        """Indica se juros compostos por sessao estao ativos."""
        return bool(self.params.get("compounding_enabled", False))

    def compounding_rate(self) -> float:
        """Retorna taxa diaria de compounding limitada ao intervalo [0, 1]."""
        rate = float(self.params.get("compounding_rate_daily", 0.0431))
        return max(0.0, min(1.0, rate))

    def calculate_session_targets(self, session_start_balance: float) -> SessionTargets:
        """Calcula banca inicial e meta win da sessao ativa."""
        start = max(0.0, float(session_start_balance))
        rate = self.compounding_rate()
        fixed = _small_account_fixed_win(self.risk_management, start)
        target = fixed if fixed is not None else start * rate
        return SessionTargets(
            session_start_balance=start,
            target_win=_round_currency(target),
            compounding_rate=rate,
        )

    def is_small_account(self, session_start_balance: float) -> bool:
        """Indica se a banca inicial usa stop win fixo de micro-conta."""
        return _small_account_fixed_win(self.risk_management, session_start_balance) is not None

    def resolve_target(
        self,
        session_start_balance: float,
        *,
        persisted_target: float | None = None,
    ) -> float:
        """Resolve meta win composta ou legado conforme configuracao."""
        if self.is_compounding_enabled():
            if persisted_target is not None and float(persisted_target) > 0.0:
                return _round_currency(float(persisted_target))
            return self.calculate_session_targets(session_start_balance).target_win
        return _resolve_legacy_stop_win_target(self.risk_management, session_start_balance)


def resolve_session_start_balance(live_balance: float, risk_management: dict[str, Any] | None) -> float:
    """Retorna banca inicial da sessao (override manual ou saldo vivo)."""
    rm = risk_management if isinstance(risk_management, dict) else {}
    params = rm.get("params") if isinstance(rm.get("params"), dict) else {}
    override = params.get("session_start_balance")
    if isinstance(override, (int, float)) and float(override) > 0.0:
        return float(override)
    return max(0.0, float(live_balance))


def _small_account_fixed_win(risk_management: dict[str, Any], session_start_balance: float) -> float | None:
    """Retorna stop win fixo quando banca inicial esta abaixo do limiar de micro-conta."""
    rm = risk_management or {}
    thr = float(rm.get("small_account_threshold", 100.0))
    if float(session_start_balance) < thr:
        return max(0.0, float(rm.get("small_account_stop_win", 10.0)))
    return None


def _resolve_legacy_stop_win_target(risk_management: dict[str, Any], session_start_balance: float) -> float:
    """Calcula meta fixa ou percentual legada quando compounding esta desativado."""
    rm = risk_management or {}
    fixed = _small_account_fixed_win(rm, session_start_balance)
    if fixed is not None:
        return fixed
    ini = float(session_start_balance)
    pct = float(rm.get("large_account_stop_win_pct", 1.0))
    pct = max(0.0, min(100.0, pct))
    return _round_currency(ini * pct / 100.0)


def resolve_stop_win_target(
    risk_management: dict[str, Any],
    session_start_balance: float,
    *,
    persisted_target: float | None = None,
) -> float:
    """Facade para resolver meta de stop win da sessao."""
    return StopWinManager(risk_management).resolve_target(
        session_start_balance,
        persisted_target=persisted_target,
    )


def resolve_max_stake_pct(
    kelly_config: dict[str, Any],
    conviction: float,
) -> float:
    """Retorna teto de stake percentual conforme conviccao."""
    base = float(kelly_config.get("max_stake_pct", 0.01))
    threshold = float(kelly_config.get("high_conviction_stake_threshold", 0.75))
    if conviction >= threshold:
        return float(kelly_config.get("max_stake_pct_high_conviction", base))
    return base


def persisted_session_target(rm: Any) -> float | None:
    """Retorna meta win persistida da sessao quando valida."""
    val = getattr(rm, "daily_stop_win_target", None)
    if isinstance(val, (int, float)) and float(val) > 0.0:
        return float(val)
    return None


def is_stop_win_reached(
    total_session_profit: float,
    target_win: float,
    *,
    risk_config: dict[str, Any] | None = None,
) -> bool:
    """Verifica se o lucro atingiu ou superou o limiar de tolerancia da meta (padrao 98%)."""
    target = float(target_win)
    profit = float(total_session_profit)
    if target <= 0.0:
        return False
    if profit + 1e-9 >= target:
        return True
    rc = risk_config or {}
    params = rc.get("params") if isinstance(rc.get("params"), dict) else {}
    tol = float(params.get("stop_win_target_tolerance_ratio", 0.98))
    tol = max(0.80, min(1.0, tol))
    return profit + 1e-9 >= target * tol
