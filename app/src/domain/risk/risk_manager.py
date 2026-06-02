"""Gerenciamento de Risco baseado no Critério de Kelly (Estratégia Profissional)."""

import datetime
import logging
import math
from typing import Any

from src.domain.risk.risk_cooldown import RiskCooldownMixin
from src.domain.risk.stop_win_target import resolve_max_stake_pct, resolve_stop_win_target


class RiskManager(RiskCooldownMixin):
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
        self._candle_interval_seconds = 900
        self._cooldown_until_mono = 0.0

        self.active_contract_ids: list[int] = []
        self.contract_to_symbol: dict[int, str] = {}
        self.cluster_results: dict[int, float] = {}
        self.expected_cluster_settlements = 0
        self.logger = logging.getLogger("AETH")

    def set_initial_bankroll(self, amount: float):
        """Define a banca inicial para rastreamento da sessão."""
        self.initial_bankroll = float(amount)

    def set_candle_interval_seconds(self, seconds: int) -> None:
        """Define duracao da vela ancora para cooldown em tempo real."""
        self._candle_interval_seconds = max(60, int(seconds))

    def reset_daily_session(self, bankroll: float) -> None:
        """Reinicia lucro de sessao e metas para novo dia (stop win diario)."""
        bal = float(bankroll)
        self.total_session_profit = 0.0
        self.initial_bankroll = bal

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

    def _apply_stop_win_aggressive_stake(
        self, bankroll: float, raw_stake: float, *, apply_stop_win: bool = True
    ) -> float:
        """Eleva stake moderadamente quando falta pouco para o stop win diario (modo ativo unico)."""
        if not apply_stop_win or not bool(self.kelly_config.get("stop_win_aggressive", False)):
            return raw_stake
        target = resolve_stop_win_target(self.config, self.initial_bankroll)
        remaining = max(0.0, target - float(self.total_session_profit))
        if remaining <= 0 or bankroll <= 0:
            return raw_stake
        payout = max(0.5, float(self.risk_params.get("payout_estimate", 0.95)))
        goal_stake = remaining / payout
        mult = max(1.0, float(self.kelly_config.get("stop_win_stake_multiplier", 1.35)))
        boosted = max(raw_stake * mult, raw_stake, goal_stake)
        cap_pct = float(self.kelly_config.get("stop_win_stake_cap_pct", 0.01))
        return min(boosted, bankroll * cap_pct)

    def stake_block_reason(
        self,
        bankroll: float,
        symbol: str,
        conviction: float = 0.5,
        *,
        apply_stop_win: bool = True,
    ) -> str | None:
        """Retorna motivo quando nenhuma stake pode ser alocada."""
        if apply_stop_win:
            target = resolve_stop_win_target(self.config, self.initial_bankroll)
            if self.total_session_profit >= target:
                return "stop_win"
        if (
            self.calculate_stake(
                bankroll,
                symbol,
                conviction=conviction,
                silent=True,
                apply_stop_win=apply_stop_win,
            )
            <= 0
        ):
            return "kelly_no_edge"
        return None

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

        b = float(self.risk_params.get("payout_estimate", 0.95))
        p = self.effective_win_rate(symbol, conviction)
        q = 1.0 - p

        kelly_f = (b * p - q) / b if b > 0 else 0.0

        loss_to_recover = sum(self.pending_loss.values())
        is_recovery_attempt = loss_to_recover > 0.0 and conviction >= self.recovery_threshold

        fractional_multiplier = float(self.kelly_config.get("fraction", 0.03))
        if self.consecutive_losses > 0 and not is_recovery_attempt:
            reduction_factor = 0.5 ** min(self.consecutive_losses, 3)
            fractional_multiplier *= reduction_factor

        f_star = max(0.0, kelly_f * fractional_multiplier)

        raw_stake = bankroll * f_star
        raw_stake = self._apply_stop_win_aggressive_stake(bankroll, raw_stake, apply_stop_win=apply_stop_win)

        if apply_stop_win:
            now_utc = datetime.datetime.now(datetime.UTC)

            in_window = 12 <= now_utc.hour < 17
            target = resolve_stop_win_target(self.config, self.initial_bankroll)
            remaining = max(0.0, target - float(self.total_session_profit))

            if in_window and conviction >= 0.75 and remaining > 0 and not self.active_contract_ids:
                goal_stake = remaining / b
                max_allowed_drawdown = bankroll * resolve_max_stake_pct(self.kelly_config, conviction)
                single_strike_stake = min(goal_stake, max_allowed_drawdown)
                if single_strike_stake > raw_stake:
                    self.logger.info(
                        "RISK: Ativando modo SINGLE STRIKE (Uma Tacada Só)! Sizing boost de $%.2f para $%.2f",
                        raw_stake,
                        single_strike_stake,
                    )
                    raw_stake = single_strike_stake

        recovery_stake = 0.0

        if is_recovery_attempt:
            needed_extra = loss_to_recover / b
            recovery_stake = needed_extra
            raw_stake += recovery_stake

        final_stake = math.ceil(raw_stake * 100) / 100 if is_recovery_attempt else math.floor(raw_stake * 100) / 100

        max_pct = resolve_max_stake_pct(self.kelly_config, conviction, is_recovery=is_recovery_attempt)
        final_stake = min(final_stake, bankroll * max_pct, self.stake_max)

        stake_min = float(self.risk_params.get("stake_min", 1.0))
        if (conviction >= 0.50 or is_recovery_attempt) and final_stake < stake_min:
            final_stake = stake_min if bankroll >= stake_min else 0.0

        cycle_id = _kwargs.get("cycle_id", 0)
        rec_info = f" | RECOVERY: ${recovery_stake:.2f}/{loss_to_recover:.2f}" if loss_to_recover > 0 else ""
        if not silent:
            self.logger.info(
                "[C%04d] KELLY: stake=$%.2f (f*=%.4f) | p=%.2f | b=%.2f | banca=$%.2f | sym=%s%s",
                int(cycle_id),
                final_stake,
                f_star,
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
            remaining_profit = profit
            for sym in list(self.pending_loss.keys()):
                if remaining_profit <= 0:
                    break
                current_loss = self.pending_loss[sym]
                if current_loss <= remaining_profit:
                    remaining_profit -= current_loss
                    self.pending_loss[sym] = 0.0
                else:
                    self.pending_loss[sym] = current_loss - remaining_profit
                    remaining_profit = 0.0

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
            self.logger.info(
                "RISK: Ciclo negativo (P&L: $%.2f) consecutive_losses=%d",
                cluster_profit,
                self.consecutive_losses,
            )
        else:
            if self.consecutive_losses > 0:
                self.logger.info(
                    "RISK: Ciclo positivo (P&L: $%.2f). Reset perdas consecutivas",
                    cluster_profit,
                )
            self.consecutive_losses = 0

        self._cooldown_until_mono = 0.0
        self.current_cooldown_ticks = 0
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
