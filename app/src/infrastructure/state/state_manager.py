"""Gerenciamento de estado de sessao e persistência de limites de trading."""

import logging
from pathlib import Path

from aether_paths import repo_path
from src.infrastructure.state.persistence_manager import PersistenceManager


class SessionState:
    """Classe que representa o estado de execucao e limites da sessao atual."""

    def __init__(self):
        """Inicializa as métricas da sessão com valores zerados."""
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.daily_stop_win_target = 0.0
        self.total_trades_today = 0
        self.stop_win_triggered = False
        self.day_key = 0

    @property
    def initial_session_balance(self) -> float:
        """Alias de compatibilidade DDD para o saldo inicial da sessão."""
        return self.initial_balance

    @initial_session_balance.setter
    def initial_session_balance(self, value: float):
        """Define o saldo inicial da sessão."""
        self.initial_balance = value

    @property
    def real_current_balance(self) -> float:
        """Alias de compatibilidade DDD para o saldo atual real da sessão."""
        return self.current_balance

    @real_current_balance.setter
    def real_current_balance(self, value: float):
        """Define o saldo atual real da sessão."""
        self.current_balance = value

    @property
    def session_profit(self) -> float:
        """Retorna o lucro acumulado da sessão atual."""
        return self.current_balance - self.initial_balance


class StateManager:
    """Gerencia a persistência física do estado da sessão e validações de limites."""

    def __init__(self, file_path: str | Path | None = None):
        """Inicializa o gerenciador com persistência e estado limpo."""
        actual_path = file_path if file_path is not None else repo_path("data", "session_state.json")
        self.persistence = PersistenceManager(actual_path)
        self.state = SessionState()
        self.logger = logging.getLogger("AETH")

    def reset_daily_metrics(self, balance: float, target: float, day_key: int):
        """Reinicia as métricas diárias da sessão para um novo dia."""
        self.state.initial_balance = float(balance)
        self.state.current_balance = float(balance)
        self.state.daily_stop_win_target = float(target)
        self.state.total_trades_today = 0
        self.state.stop_win_triggered = False
        self.state.day_key = int(day_key)
        self.save_state()

    def reset_daily_session_metrics(self, balance: float, target: float, day_key: int):
        """Alias de compatibilidade DDD para reset_daily_metrics."""
        self.reset_daily_metrics(balance, target, day_key)

    def check_session_limits(self) -> bool:
        """Valida se os limites financeiros (Stop Win) foram atingidos na sessão.

        Garante que o Stop Win só seja ativado se houver lucro real E ao menos
        uma operação (trade) realizada na sessão atual para evitar falso gatilho no boot.
        """
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
        """Carrega os dados persistidos do estado da sessão do arquivo JSON."""
        data = self.persistence.load()
        if data:
            self.state.initial_balance = float(data.get("initial_balance", 0.0))
            self.state.current_balance = float(data.get("current_balance", 0.0))
            self.state.daily_stop_win_target = float(data.get("daily_stop_win_target", 0.0))
            self.state.total_trades_today = int(data.get("total_trades_today", 0))
            self.state.stop_win_triggered = bool(data.get("stop_win_triggered", False))
            self.state.day_key = int(data.get("day_key", 0))
            return True
        return False

    def save_state(self):
        """Grava o estado atual da sessão no arquivo JSON persistido."""
        data = {
            "initial_balance": self.state.initial_balance,
            "current_balance": self.state.current_balance,
            "daily_stop_win_target": self.state.daily_stop_win_target,
            "total_trades_today": self.state.total_trades_today,
            "stop_win_triggered": self.state.stop_win_triggered,
            "day_key": self.state.day_key,
        }
        self.persistence.save(data)
