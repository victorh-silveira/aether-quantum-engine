"""Gerenciamento de Risco baseado no Critério de Kelly (Estratégia Profissional)."""

import logging
from typing import Any

from src.domain.risk.bayesian_win_rate import bayesian_win_rate
from src.domain.risk.recovery_conviction import recovery_dl_conviction_ok, recovery_dl_entry_allowed
from src.domain.risk.recovery_hurst_gate import resolve_recovery_signal_floor
from src.domain.risk.risk_cluster import finalize_risk_cluster
from src.domain.risk.risk_contract_result import apply_contract_settlement_result
from src.domain.risk.risk_cooldown import RiskCooldownMixin
from src.domain.risk.risk_manager_restore import apply_risk_snapshot, build_risk_state_snapshot
from src.domain.risk.risk_proposal_skip import ProposalSkipMixin
from src.domain.risk.risk_recovery_state import (
    cointegration_redirect_armed,
    micro_tail_stake_cap,
    pending_loss_total as sum_pending_loss,
    recovery_financially_active as is_recovery_financially_active,
)
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager
from src.domain.risk.soft_recovery_policy import cointegration_valve_suppressed, resolve_soft_recovery_config
from src.domain.risk.stake_target_proximity import apply_target_proximity_damping
from src.domain.risk.stop_win_target import persisted_session_target, resolve_stop_win_target
from src.domain.risk.symbol_loss_cooldown import SymbolLossCooldownMixin


class RiskManager(RiskCooldownMixin, SymbolLossCooldownMixin, ProposalSkipMixin):
    """Gerenciador de Risco que utiliza a fórmula de Kelly para dimensionamento de posição."""

    def __init__(self, config: dict[str, Any]):
        """Inicializa o RiskManager com configurações de Kelly."""
        self.config = config
        self.kelly_config = config.get("kelly", {})
        self.soft_recovery_config = resolve_soft_recovery_config(config)
        legacy = config.get("dlambert", {}) if isinstance(config.get("dlambert"), dict) else {}
        self.dlambert_config = {
            **legacy,
            "dlambert_enabled": bool(self.soft_recovery_config.get("enabled", True)),
            "soft_recovery": dict(self.soft_recovery_config),
        }
        self.risk_params = config.get("params", {"payout_estimate": 0.95, "stake_min": 1.0})
        self.limits = config.get("limits", {})

        self._rolling_wins: dict[str, list[int]] = {}
        self.initial_bankroll = 0.0
        self.daily_stop_win_target = 0.0
        self.total_session_profit = 0.0
        self.last_result_tick = 0
        self.base_cooldown = self.risk_params.get("entry_cooldown_ticks", 0)
        self.current_cooldown_ticks = self.base_cooldown
        self._last_entry_conviction = 0.0
        self.consecutive_losses_linear = 0
        self.dlambert_unit = 0.0
        self.pending_loss: dict[str, float] = {}
        self.last_loss_stake = 0.0
        self.proposal_skip_cycles: dict[str, int] = {}
        self.contract_stakes: dict[int, float] = {}
        self.contract_requested_stakes: dict[int, float] = {}
        self.init_symbol_loss_cooldown()
        self._candle_interval_seconds = 60
        self._cooldown_until_mono = 0.0

        self.active_contract_ids: list[int] = []
        self.contract_to_symbol: dict[int, str] = {}
        self.cluster_results: dict[int, float] = {}
        self.expected_cluster_settlements = 0
        self.logger = logging.getLogger("AETH")

    def set_initial_bankroll(self, amount: float):
        """Define a banca inicial para rastreamento da sessão."""
        self.initial_bankroll = float(amount)

    def pending_loss_total(self) -> float:
        """Soma drawdown financeiro pendente da sessao."""
        return sum_pending_loss(self.pending_loss)

    def recovery_financially_active(self) -> bool:
        """Indica se a sessao ainda deve operar em modo recovery financeiro."""
        return is_recovery_financially_active(self.pending_loss)

    def cointegration_redirect_active(self) -> bool:
        """True quando Consensus Cointegration Redirect deve desviar o soft recovery."""
        bankroll = float(self.initial_bankroll) if self.initial_bankroll > 0.0 else 100.0
        pending = self.pending_loss_total()
        if cointegration_valve_suppressed(bankroll, pending, soft_recovery=self.soft_recovery_config):
            return False
        threshold = self.soft_recovery_config.get("coing_redirect_drawdown_threshold")
        return cointegration_redirect_armed(
            bankroll,
            pending,
            threshold=float(threshold) if threshold is not None else None,
        )

    def max_safe_tail_cap(self, bankroll: float | None = None) -> float:
        """Retorna teto de cauda 4.2*U para micro-banca a partir do nivel linear 4."""
        bal = float(bankroll) if bankroll is not None else float(self.initial_bankroll)
        return micro_tail_stake_cap(bal if bal > 0.0 else 100.0)

    def set_candle_interval_seconds(self, seconds: int) -> None:
        """Define duracao da vela ancora para cooldown em tempo real."""
        self._candle_interval_seconds = max(1, int(seconds))

    def reset_session(self, bankroll: float, *, target: float = 0.0) -> None:
        """Reinicia lucro de sessao e meta de stop win da instancia corrente."""
        bal = float(bankroll)
        self.total_session_profit = 0.0
        self.initial_bankroll = bal
        self.daily_stop_win_target = max(0.0, float(target))
        self.dlambert_unit = 0.0
        self.consecutive_losses_linear = 0
        self.pending_loss.clear()

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

    def effective_win_rate(
        self,
        symbol: str,
        conviction: float = 0.5,
        metrics: dict[str, Any] | None = None,
    ) -> float:
        """Define p Kelly via bayesian_win_rate (live_wr/Brier/Z ou rolling interno)."""
        base_p = float(conviction)
        if not self.kelly_config.get("dynamic_win_rate", False):
            return base_p
        min_s = int(self.kelly_config.get("dynamic_min_samples", 20))
        wr, n = self.get_wr_rolling_stats(symbol)
        return bayesian_win_rate(
            base_p,
            rolling_wr=wr,
            rolling_n=n,
            metrics=metrics,
            dynamic_min_samples=min_s,
        )

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
            target = resolve_stop_win_target(
                self.config,
                self.initial_bankroll,
                persisted_target=persisted_session_target(self),
            )
            if self.total_session_profit >= target:
                return "stop_win"

        stake = self.calculate_stake(
            bankroll,
            symbol,
            conviction=conviction,
            silent=True,
            apply_stop_win=apply_stop_win,
            **kwargs,
        )
        if stake > 0:
            return None
        stake_min = float(self.risk_params.get("stake_min", 1.0))
        if float(bankroll) + 1e-12 < stake_min:
            return "bankroll_below_stake_min"
        return None

    def _recovery_dl_conviction_ok(self, dl_metrics: dict) -> bool:
        """Exige piso de sinal e val_accuracy para recovery com metricas DL."""
        return recovery_dl_conviction_ok(
            dl_metrics,
            self.kelly_config,
            self.dlambert_config,
            pending_loss=self.pending_loss,
            consecutive_losses_linear=int(getattr(self, "consecutive_losses_linear", 0)),
        )

    def _recovery_allowed(self, _symbol: str, _conviction: float, **kwargs) -> bool:
        """Recovery D'Alembert ativo com perda pendente e conviccao minima nas metricas DL."""
        if not self.recovery_financially_active():
            return False
        dl_metrics = kwargs.get("dl_metrics")
        if isinstance(dl_metrics, dict):
            return recovery_dl_entry_allowed(
                dl_metrics,
                self.kelly_config,
                self.dlambert_config,
                pending_loss=self.pending_loss,
                consecutive_losses_linear=int(getattr(self, "consecutive_losses_linear", 0)),
                recovery_forced=bool(kwargs.get("recovery_forced")),
            )
        return True

    def apply_kelly_target_proximity_damping(
        self,
        kelly_stake_raw: float,
        *,
        target_win: float | None = None,
    ) -> float:
        """Comprime stake Kelly bruta conforme distancia da meta de stop win da sessao."""
        target = (
            float(target_win)
            if target_win is not None
            else resolve_stop_win_target(
                self.config,
                self.initial_bankroll,
                persisted_target=persisted_session_target(self),
            )
        )
        return apply_target_proximity_damping(kelly_stake_raw, target, self.total_session_profit)

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

    def record_contract_stake(self, contract_id: int, stake: float, *, requested: float | None = None) -> None:
        """Associa stake enviada ao contrato para progressao D'Alembert."""
        cid = int(contract_id)
        self.contract_stakes[cid] = float(stake)
        planned = float(requested) if requested is not None else float(stake)
        self.contract_requested_stakes[cid] = planned

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
        apply_contract_settlement_result(
            self,
            profit,
            contract_id,
            symbol,
            current_tick,
            direction=direction,
        )

    def _finalize_cluster(self):
        """Limpeza após ciclo."""
        finalize_risk_cluster(self)

    def get_state(self) -> dict[str, Any]:
        """Estado para persistência."""
        return build_risk_state_snapshot(self)

    def restore_state(self, data: dict[str, Any]) -> None:
        """Restaura campos de risco a partir de snapshot persistido."""
        apply_risk_snapshot(self, data)

    def recovery_signal_floor(self, hurst: float, *, recovery_skip_counter: int = 0) -> float:
        """Piso de sinal em recovery ajustado logaritmicamente pelo Hurst do candidato."""
        return resolve_recovery_signal_floor(
            self.kelly_config,
            hurst=hurst,
            consecutive_losses=self.consecutive_losses_linear,
            total_session_profit=self.total_session_profit,
            recovery_skip_counter=recovery_skip_counter,
        )
