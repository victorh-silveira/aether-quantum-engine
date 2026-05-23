"""Gerenciamento de Risco baseado no Critério de Kelly (Estratégia Profissional)."""

import logging
import math
from typing import Any

from src.domain.risk.entry_cooldown import resolve_entry_cooldown_ticks
from src.domain.risk.stop_win_target import resolve_stop_win_target


class RiskManager:
    """Gerenciador de Risco que utiliza a fórmula de Kelly para dimensionamento de posição."""

    def __init__(self, config: dict[str, Any]):
        """Inicializa o RiskManager com configurações de Kelly."""
        self.config = config
        self.kelly_config = config.get("kelly", {})
        self.risk_params = config.get("params", {"payout_estimate": 0.95, "stake_min": 1.0, "stake_max": 1000000.0})
        self.limits = config.get("limits", {})
        self.stake_max = float(self.risk_params.get("stake_max", 1000000.0))

        self._rolling_wins: dict[str, list[int]] = {}
        self.initial_bankroll = 0.0
        self.total_session_profit = 0.0
        self.last_result_tick = 0
        self.base_cooldown = self.risk_params.get("entry_cooldown_ticks", 0)
        self.current_cooldown_ticks = self.base_cooldown
        self._last_entry_conviction = 0.0
        self.consecutive_losses = 0
        self.pending_loss: dict[str, float] = {}
        self.recovery_threshold = float(self.kelly_config.get("recovery_conviction_threshold", 0.60))
        self.session_max_drawdown_pct = float(
            self.kelly_config.get("session_max_drawdown_pct", self.config.get("session_max_drawdown_pct", 15.0))
        )
        self.peak_bankroll = 0.0

        self.active_contract_ids: list[int] = []
        self.contract_to_symbol: dict[int, str] = {}
        self.cluster_results: dict[int, float] = {}
        self.expected_cluster_settlements = 0
        self.logger = logging.getLogger("AETH")

    def set_initial_bankroll(self, amount: float):
        """Define a banca inicial para rastreamento da sessão."""
        self.initial_bankroll = float(amount)
        self.peak_bankroll = float(amount)

    def reset_daily_session(self, bankroll: float) -> None:
        """Reinicia lucro de sessao e metas para novo dia (stop win diario)."""
        bal = float(bankroll)
        self.total_session_profit = 0.0
        self.initial_bankroll = bal
        self.peak_bankroll = bal

    def record_trade_outcome(self, symbol: str, *, won: bool) -> None:
        """Registra o resultado para cálculo de win rate dinâmico."""
        lst = self._rolling_wins.setdefault(str(symbol), [])
        lst.append(1 if won else 0)
        cap = 100
        if len(lst) > cap:
            del lst[: len(lst) - cap]

    def get_wr_rolling_stats(self, symbol: str) -> tuple[float | None, int]:
        """Retorna o win rate atual do símbolo."""
        lst = self._rolling_wins.get(str(symbol), [])
        n = len(lst)
        if n == 0:
            return None, 0
        return float(sum(lst)) / float(n), n

    def effective_win_rate(self, symbol: str, conviction: float = 0.5) -> float:
        """Define a probabilidade (p) baseada na convicção da IA ou histórico."""
        base_p = conviction
        if not self.kelly_config.get("dynamic_win_rate", False):
            return base_p

        min_s = int(self.kelly_config.get("dynamic_min_samples", 20))
        wr, n = self.get_wr_rolling_stats(symbol)

        if wr is not None and n >= min_s:
            return (base_p * 0.7) + (wr * 0.3)

        return base_p

    def calculate_stake(
        self,
        bankroll: float,
        symbol: str,
        conviction: float = 0.5,
        *,
        silent: bool = False,
        apply_stop_win: bool = True,
        **_kwargs,
    ) -> float:
        """Calcula a stake usando o Critério de Kelly com trava de Stop Win."""
        if apply_stop_win:
            target = resolve_stop_win_target(self.config, self.initial_bankroll)
            if self.total_session_profit >= target:
                self.logger.info(f"STOP WIN: Meta de ${target:.2f} atingida. Encerrando operações do dia.")
                return 0.0

        peak = max(self.peak_bankroll, self.initial_bankroll, float(bankroll))
        self.peak_bankroll = peak
        max_dd = max(0.0, float(self.session_max_drawdown_pct))
        if max_dd > 0 and peak > 0:
            dd_pct = ((peak - float(bankroll)) / peak) * 100.0
            if dd_pct >= max_dd:
                if not silent:
                    self.logger.info(
                        "DRAWDOWN_BRAKE: pausa (dd=%.2f%% >= %.2f%%) banca=$%.2f pico=$%.2f",
                        dd_pct,
                        max_dd,
                        bankroll,
                        peak,
                    )
                return 0.0

        b = float(self.risk_params.get("payout_estimate", 0.95))
        p = self.effective_win_rate(symbol, conviction)
        q = 1.0 - p

        kelly_f = (b * p - q) / b if b > 0 else 0.0

        loss_to_recover = self.pending_loss.get(symbol, 0.0)
        is_recovery_attempt = loss_to_recover > 0.0 and conviction >= self.recovery_threshold

        fractional_multiplier = float(self.kelly_config.get("fraction", 0.1))
        if self.consecutive_losses > 0 and loss_to_recover == 0.0:
            reduction_factor = 0.5 ** min(self.consecutive_losses, 3)
            fractional_multiplier *= reduction_factor

        f_star = max(0.0, kelly_f * fractional_multiplier)

        max_pct = float(self.kelly_config.get("max_stake_pct", 0.05))
        stake_thr = float(self.kelly_config.get("high_conviction_stake_threshold", 0.85))
        if float(conviction) >= stake_thr:
            max_pct = max(
                max_pct,
                float(self.kelly_config.get("max_stake_pct_high_conviction", max_pct)),
            )
        base_f_star = min(f_star, max_pct)
        raw_stake = bankroll * base_f_star

        recovery_stake = 0.0

        if is_recovery_attempt:
            needed_extra = loss_to_recover / b

            max_recovery_pct = float(self.kelly_config.get("max_recovery_stake_pct", 0.15))

            potential_total = raw_stake + needed_extra
            if potential_total / bankroll > max_recovery_pct:
                recovery_stake = (bankroll * max_recovery_pct) - raw_stake
                recovery_stake = max(0.0, recovery_stake)
            else:
                recovery_stake = needed_extra

            raw_stake += recovery_stake

        final_stake = math.ceil(raw_stake * 100) / 100 if is_recovery_attempt else math.floor(raw_stake * 100) / 100

        stake_min = float(self.risk_params.get("stake_min", 1.0))
        if (conviction >= 0.50 or is_recovery_attempt) and final_stake < stake_min:
            final_stake = stake_min if bankroll >= stake_min else 0.0

        final_stake = min(final_stake, self.stake_max)

        cycle_id = _kwargs.get("cycle_id", 0)
        rec_info = f" | RECOVERY: ${recovery_stake:.2f}/{loss_to_recover:.2f}" if loss_to_recover > 0 else ""
        if not silent:
            self.logger.info(
                "[C%04d] KELLY: stake=$%.2f (f*=%.4f) | p=%.2f | b=%.2f | banca=$%.2f | sym=%s%s",
                int(cycle_id),
                final_stake,
                base_f_star,
                p,
                b,
                bankroll,
                symbol,
                rec_info,
            )

        return final_stake

    def begin_cluster(self, expected_settlements: int) -> None:
        """Marca inicio de cluster e quantidade de liquidacoes esperadas."""
        self.expected_cluster_settlements = max(0, int(expected_settlements))
        self.cluster_results = {}

    def register_result(self, profit: float, contract_id: int, symbol: str, current_tick: int = 0):
        """Registra lucro/prejuízo e atualiza estatísticas."""
        if contract_id not in self.active_contract_ids:
            return

        self.cluster_results[contract_id] = profit
        self.total_session_profit += profit
        self.last_result_tick = current_tick
        self.record_trade_outcome(symbol, won=profit >= 0.0)

        if profit < 0:
            self.pending_loss[symbol] = self.pending_loss.get(symbol, 0.0) + abs(profit)
        else:
            current_loss = self.pending_loss.get(symbol, 0.0)
            self.pending_loss[symbol] = max(0.0, current_loss - profit)

        self.active_contract_ids = [x for x in self.active_contract_ids if int(x) != int(contract_id)]

        expected = self.expected_cluster_settlements
        if (
            expected > 0
            and len(self.cluster_results) >= expected
            or not self.active_contract_ids
            and self.cluster_results
        ):
            self._finalize_cluster()

    def _finalize_cluster(self):
        """Limpeza após ciclo."""
        cluster_profit = sum(self.cluster_results.values())
        if cluster_profit < 0.0:
            self.consecutive_losses += 1
            multiplier = 2 ** min(self.consecutive_losses, 4)
            self.current_cooldown_ticks = int(self.base_cooldown * multiplier)
            self.logger.info(
                "RISK: Ciclo negativo (P&L: $%.2f). Aumentando cooldown para %d ticks (consecutive_losses=%d)",
                cluster_profit,
                self.current_cooldown_ticks,
                self.consecutive_losses,
            )
        else:
            if self.consecutive_losses > 0:
                self.logger.info(
                    "RISK: Ciclo positivo (P&L: $%.2f). Resetando cooldown para %d ticks",
                    cluster_profit,
                    self.base_cooldown,
                )
            self.consecutive_losses = 0
            self.current_cooldown_ticks = self.base_cooldown

        self.active_contract_ids = []
        self.contract_to_symbol = {}
        self.cluster_results = {}
        self.expected_cluster_settlements = 0

    def get_state(self) -> dict[str, Any]:
        """Estado para persistência."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "total_session_profit": self.total_session_profit,
            "last_result_tick": self.last_result_tick,
            "rolling_wins": {k: list(v) for k, v in self._rolling_wins.items()},
            "pending_loss": dict(self.pending_loss),
            "consecutive_losses": self.consecutive_losses,
            "current_cooldown_ticks": self.current_cooldown_ticks,
        }

    def register_entry_conviction(self, conviction: float) -> None:
        """Registra conviccao da ultima entrada para cooldown dinamico."""
        self._last_entry_conviction = max(0.0, float(conviction))

    def effective_cooldown_ticks(self) -> int:
        """Cooldown efetivo apos cluster (dinamico por conviccao da ultima entrada)."""
        target = resolve_entry_cooldown_ticks(self.config, self._last_entry_conviction)
        active = int(self.current_cooldown_ticks)
        if target <= 0:
            return active
        if active <= 0:
            return target
        return min(active, target)

    def is_on_cooldown(self, current_tick: int) -> bool:
        """Cooldown entre operações."""
        need = self.effective_cooldown_ticks()
        if self.last_result_tick == 0 or need == 0:
            return False
        elapsed = current_tick - self.last_result_tick
        return elapsed < need
