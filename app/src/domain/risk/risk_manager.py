"""Gerenciamento de Risco baseado no Critério de Kelly (Estratégia Profissional)."""

import logging
from typing import Any

from src.domain.risk.martingale_gate import apply_win_to_pending_loss, martingale_allowed
from src.domain.risk.risk_cluster import finalize_risk_cluster
from src.domain.risk.risk_cooldown import RiskCooldownMixin
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager
from src.domain.risk.stop_win_target import resolve_stop_win_target
from src.domain.risk.symbol_loss_cooldown import SymbolLossCooldownMixin


class RiskManager(RiskCooldownMixin, SymbolLossCooldownMixin):
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
        self.last_martingale_stake = 0.0
        self.last_loss_stake = 0.0
        self._prev_martingale_stake = 0.0
        self.contract_stakes: dict[int, float] = {}
        self.init_symbol_loss_cooldown()
        self.martingale_native = bool(self.kelly_config.get("martingale_native", True))
        self.martingale_block_repeat_loss = bool(self.kelly_config.get("martingale_block_repeat_loss", False))
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
        **kwargs,
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
                **kwargs,
            )
            <= 0
        ):
            return "kelly_no_edge"
        return None

    def _martingale_allowed(self, symbol: str, conviction: float, **kwargs) -> bool:
        """Martingale ativo sempre que houver perda pendente (modo nativo)."""
        _ = (conviction, kwargs)
        return martingale_allowed(
            pending_loss=self.pending_loss,
            martingale_native=self.martingale_native,
            block_repeat_loss=self.martingale_block_repeat_loss,
            symbol=symbol,
            order_direction=kwargs.get("order_direction"),
            last_loss_symbol=self.last_loss_symbol,
            last_loss_direction=self.last_loss_direction,
        )

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
        return calculate_stake_for_manager(
            self,
            bankroll,
            symbol,
            conviction,
            silent=silent,
            apply_stop_win=apply_stop_win,
            kwargs=_kwargs,
        )

    def begin_cluster(self, expected_settlements: int) -> None:
        """Marca inicio de cluster e quantidade de liquidacoes esperadas."""
        self.expected_cluster_settlements = max(0, int(expected_settlements))
        self.cluster_results = {}

    def record_contract_stake(self, contract_id: int, stake: float) -> None:
        """Associa stake enviada ao contrato para progressao martingale."""
        self.contract_stakes[int(contract_id)] = float(stake)

    def register_result(
        self,
        profit: float,
        contract_id: int,
        symbol: str,
        current_tick: int = 0,
        *,
        direction: str | None = None,
    ):
        """Registra lucro/prejuízo e atualiza estatísticas."""
        if contract_id not in self.active_contract_ids:
            return

        self.contract_stakes.pop(int(contract_id), None)
        self.cluster_results[contract_id] = profit
        self.total_session_profit += profit
        self.last_result_tick = current_tick
        self.record_trade_outcome(symbol, won=profit >= 0.0)

        if profit < 0:
            loss_amt = abs(profit)
            self.pending_loss[symbol] = self.pending_loss.get(symbol, 0.0) + loss_amt
            self.last_loss_stake = self.contract_stakes.pop(int(contract_id), loss_amt)
            self.register_symbol_loss_cooldown(symbol, direction=direction)
        else:
            apply_win_to_pending_loss(self.pending_loss, profit)
            if sum(self.pending_loss.values()) <= 0.0:
                self.last_martingale_stake = 0.0
                self.last_loss_stake = 0.0

        self.active_contract_ids = [x for x in self.active_contract_ids if int(x) != int(contract_id)]

        expected = self.expected_cluster_settlements
        cluster_done = expected > 0 and len(self.cluster_results) >= expected
        idle_done = not self.active_contract_ids and self.cluster_results
        if cluster_done or idle_done:
            self._finalize_cluster()

    def _finalize_cluster(self):
        """Limpeza após ciclo."""
        finalize_risk_cluster(self)

    def get_state(self) -> dict[str, Any]:
        """Estado para persistência."""
        return {
            "initial_bankroll": self.initial_bankroll,
            "total_session_profit": self.total_session_profit,
            "last_result_tick": self.last_result_tick,
            "rolling_wins": {k: list(v) for k, v in self._rolling_wins.items()},
            "pending_loss": dict(self.pending_loss),
            "last_martingale_stake": self.last_martingale_stake,
            "last_loss_stake": self.last_loss_stake,
            "consecutive_losses": self.consecutive_losses,
            "current_cooldown_ticks": self.current_cooldown_ticks,
            **self.symbol_cooldown_state(),
        }
