"""Gerenciamento de estado de sessao e persistencia de limites de trading."""

import logging
from pathlib import Path

from aether_paths import repo_path
from src.infrastructure.state.persistence_manager import PersistenceManager


class SessionState:
    """Classe que representa o estado de execucao e limites da sessao atual."""

    def __init__(self):
        """Inicializa as metricas da sessao com valores zerados."""
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.daily_stop_win_target = 0.0
        self.total_trades_today = 0
        self.stop_win_triggered = False

    @property
    def session_start_balance(self) -> float:
        """Banca inicial da sessao ativa."""
        return self.initial_balance

    @session_start_balance.setter
    def session_start_balance(self, value: float):
        """Define banca inicial da sessao ativa."""
        self.initial_balance = float(value)

    @property
    def initial_session_balance(self) -> float:
        """Alias legado para banca inicial da sessao."""
        return self.initial_balance

    @initial_session_balance.setter
    def initial_session_balance(self, value: float):
        """Alias legado setter para banca inicial da sessao."""
        self.initial_balance = value

    @property
    def real_current_balance(self) -> float:
        """Saldo corrente da sessao."""
        return self.current_balance

    @real_current_balance.setter
    def real_current_balance(self, value: float):
        """Atualiza saldo corrente da sessao."""
        self.current_balance = value

    @property
    def session_profit(self) -> float:
        """Lucro acumulado da sessao ativa."""
        return self.current_balance - self.initial_balance


class StateManager:
    """Gerencia a persistencia fisica do estado da sessao e validacoes de limites."""

    def __init__(self, file_path: str | Path | None = None):
        actual_path = file_path if file_path is not None else repo_path("data", "session_state.json")
        self.persistence = PersistenceManager(actual_path)
        self.state = SessionState()
        self.logger = logging.getLogger("AETH")

    def reset_session_metrics(self, balance: float, target: float):
        """Reinicia metricas da sessao ativa corrente."""
        self.state.initial_balance = float(balance)
        self.state.current_balance = float(balance)
        self.state.daily_stop_win_target = float(target)
        self.state.total_trades_today = 0
        self.state.stop_win_triggered = False
        self.save_state()

    def reset_daily_metrics(self, balance: float, target: float, day_key: int = 0, *, max_loss: float = 0.0):
        """Alias legado: reinicia metricas de sessao sem day_key."""
        _ = day_key, max_loss
        self.reset_session_metrics(balance, target)

    def reset_daily_session_metrics(self, balance: float, target: float, day_key: int = 0, *, max_loss: float = 0.0):
        """Alias legado para reset_session_metrics."""
        self.reset_daily_metrics(balance, target, day_key, max_loss=max_loss)

    def check_session_limits(self) -> bool:
        """True quando o lucro da sessao atinge a meta de stop win."""
        daily_profit = self.state.session_profit
        if (
            self.state.daily_stop_win_target > 0.0
            and daily_profit >= self.state.daily_stop_win_target
            and self.state.total_trades_today > 0
        ):
            self.state.stop_win_triggered = True
        else:
            self.state.stop_win_triggered = False
        return self.state.stop_win_triggered

    def load_state(self) -> bool:
        """Carrega metricas da sessao do disco."""
        data = self.persistence.load()
        if data:
            self.state.initial_balance = float(data.get("initial_balance", 0.0))
            self.state.current_balance = float(data.get("current_balance", 0.0))
            self.state.daily_stop_win_target = float(data.get("daily_stop_win_target", 0.0))
            self.state.total_trades_today = int(data.get("total_trades_today", 0))
            self.state.stop_win_triggered = bool(data.get("stop_win_triggered", False))
            return True
        return False

    def save_state(self):
        """Persiste metricas da sessao em disco."""
        data = {
            "initial_balance": self.state.initial_balance,
            "current_balance": self.state.current_balance,
            "daily_stop_win_target": self.state.daily_stop_win_target,
            "total_trades_today": self.state.total_trades_today,
            "stop_win_triggered": self.state.stop_win_triggered,
        }
        self.persistence.save(data)
