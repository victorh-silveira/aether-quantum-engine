"""Gerenciamento de Risco baseado no Critério de Kelly (Estratégia Profissional)."""

import logging
from typing import Any

from src.domain.risk.martingale_gate import apply_win_to_pending_loss, martingale_allowed
from src.domain.risk.risk_cluster import finalize_risk_cluster
from src.domain.risk.risk_cooldown import RiskCooldownMixin
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager
from src.domain.risk.stake_sizing import raw_side_from_metrics
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
        self.recovery_symbol_loss_streak: dict[str, int] = {}
        self.proposal_skip_cycles: dict[str, int] = {}
        self.contract_stakes: dict[int, float] = {}
        self.init_symbol_loss_cooldown()
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
        self._candle_interval_seconds = max(1, int(seconds))

    def register_proposal_failure(self, symbol: str, *, cycles: int = 6) -> None:
        """Marca simbolo para pular selecao apos falha de proposta na Deriv."""
        hold = max(int(cycles), int(self.proposal_skip_cycles.get(str(symbol), 0)))
        self.proposal_skip_cycles[str(symbol)] = hold

    def decay_proposal_skip_cycles(self) -> None:
        """Reduz cooldown de simbolos com proposta rejeitada."""
        expired: list[str] = []
        for symbol, remaining in self.proposal_skip_cycles.items():
            left = int(remaining) - 1
            if left <= 0:
                expired.append(symbol)
            else:
                self.proposal_skip_cycles[symbol] = left
        for symbol in expired:
            self.proposal_skip_cycles.pop(symbol, None)

    def proposal_skip_symbols(self) -> frozenset[str]:
        """Simbolos temporariamente excluidos apos falha de proposta."""
        return frozenset(s for s, n in self.proposal_skip_cycles.items() if int(n) > 0)

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
        if 0.50 <= base_p <= 0.55:
            base_p = 0.56 + (base_p - 0.50) * 0.6

        if not self.kelly_config.get("dynamic_win_rate", False):
            return base_p

        min_s = int(self.kelly_config.get("dynamic_min_samples", 20))
        wr, n = self.get_wr_rolling_stats(symbol)

        if wr is not None and n >= min_s:
            return (base_p * 0.7) + (wr * 0.3)

        return base_p

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

    def _martingale_dl_conviction_ok(self, dl_metrics: dict) -> bool:
        """Exige piso de sinal e val_accuracy para martingale com metricas DL."""
        if dl_metrics.get("deploy_ok") is False:
            return False
        min_conv = float(self.kelly_config.get("martingale_sizing_conviction", 0.58))
        pending = sum(float(v) for v in self.pending_loss.values())
        force_min = float(self.kelly_config.get("recovery_martingale_min_conviction", min_conv))
        force_pending = float(self.kelly_config.get("recovery_force_pending_min", 0.0))
        if force_pending > 0.0 and pending + 1e-9 >= force_pending:
            min_conv = min(min_conv, force_min)
        min_val = float(self.kelly_config.get("martingale_min_val_accuracy", 0.50))
        score = float(dl_metrics.get("trade_score", dl_metrics.get("conviction", 0.0)))
        raw_side = raw_side_from_metrics(dl_metrics)
        val = float(dl_metrics.get("val_accuracy", 0.0))
        if min_val > 0.0 and val + 1e-9 < min_val:
            return False
        if max(score, raw_side) + 1e-9 >= min_conv:
            return True
        return score < 1e-9 and raw_side + 1e-9 >= min_conv

    def _martingale_allowed(self, _symbol: str, _conviction: float, **kwargs) -> bool:
        """Martingale ativo com perda pendente e conviccao minima nas metricas DL."""
        if not martingale_allowed(pending_loss=self.pending_loss):
            return False
        dl_metrics = kwargs.get("dl_metrics")
        if isinstance(dl_metrics, dict):
            if dl_metrics.get("deploy_ok") is False:
                return False
            pending = sum(float(v) for v in self.pending_loss.values())
            if pending > 0.0 and bool(self.kelly_config.get("recovery_martingale_always", True)):
                return True
            min_val = float(self.kelly_config.get("martingale_min_val_accuracy", 0.50))
            val = float(dl_metrics.get("val_accuracy", 0.0))
            if min_val > 0.0 and val + 1e-9 < min_val:
                return False
            return self._martingale_dl_conviction_ok(dl_metrics)
        return True

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
        if contract_id in self.cluster_results:
            return

        tracked = int(contract_id) in self.active_contract_ids
        late = not tracked and int(contract_id) in self.contract_to_symbol
        if not tracked and not late:
            return
        if late:
            self.logger.debug("RISK: Liquidacao tardia cid=%s aplicada ao pending.", contract_id)

        recorded_stake = self.contract_stakes.pop(int(contract_id), None)
        self.cluster_results[contract_id] = profit
        self.total_session_profit += profit
        self.last_result_tick = current_tick
        self.record_trade_outcome(symbol, won=profit >= 0.0)

        had_pending = sum(self.pending_loss.values()) > 0.0
        if profit < 0:
            loss_amt = abs(profit)
            self.pending_loss[symbol] = self.pending_loss.get(symbol, 0.0) + loss_amt
            self.last_loss_stake = float(recorded_stake) if recorded_stake else loss_amt
            self.register_symbol_loss_cooldown(symbol, direction=direction)
            if had_pending:
                streak = int(self.recovery_symbol_loss_streak.get(symbol, 0)) + 1
                self.recovery_symbol_loss_streak[symbol] = streak
        else:
            apply_win_to_pending_loss(self.pending_loss, profit)
            self.recovery_symbol_loss_streak.pop(symbol, None)
            if sum(self.pending_loss.values()) <= 0.0:
                self.last_martingale_stake = 0.0
                self.last_loss_stake = 0.0
                self.recovery_symbol_loss_streak = {}

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
            "recovery_symbol_loss_streak": dict(self.recovery_symbol_loss_streak),
            **self.symbol_cooldown_state(),
        }
