from unittest.mock import patch

from src.infrastructure.state.state_manager import SessionState, StateManager


def test_session_state_properties():
    """Testa os getters, setters e propriedades de SessionState."""
    state = SessionState()
    assert state.initial_session_balance == 0.0
    assert state.real_current_balance == 0.0
    assert state.session_profit == 0.0

    state.initial_session_balance = 1000.0
    state.real_current_balance = 1050.0
    assert state.initial_balance == 1000.0
    assert state.current_balance == 1050.0
    assert state.session_profit == 50.0


def test_state_manager_init_and_reset(tmp_path):
    """Testa inicialização e reset de métricas no StateManager."""
    file_path = tmp_path / "test_session_state.json"
    mgr = StateManager(file_path=file_path)
    assert mgr.state.initial_balance == 0.0

    mgr.reset_daily_session_metrics(1500.0, 50.0, 12345)
    assert mgr.state.initial_balance == 1500.0
    assert mgr.state.current_balance == 1500.0
    assert mgr.state.daily_stop_win_target == 50.0
    assert mgr.state.day_key == 12345
    assert mgr.state.total_trades_today == 0
    assert mgr.state.stop_win_triggered is False


def test_state_manager_check_session_limits(tmp_path):
    """Testa a validação dos limites da sessão (Stop Win)."""
    file_path = tmp_path / "test_session_state.json"
    mgr = StateManager(file_path=file_path)

    assert mgr.check_session_limits() is False

    mgr.state.daily_stop_win_target = 50.0
    mgr.state.initial_balance = 1000.0
    mgr.state.current_balance = 1060.0
    mgr.state.total_trades_today = 0
    assert mgr.check_session_limits() is False

    mgr.state.total_trades_today = 1
    assert mgr.check_session_limits() is True

    mgr.state.current_balance = 1040.0
    assert mgr.check_session_limits() is False


def test_state_manager_save_load_cycle(tmp_path):
    """Testa o ciclo de salvar e carregar o estado."""
    file_path = tmp_path / "test_session_state.json"
    mgr = StateManager(file_path=file_path)

    mgr.state.initial_balance = 2000.0
    mgr.state.current_balance = 2100.0
    mgr.state.daily_stop_win_target = 100.0
    mgr.state.total_trades_today = 3
    mgr.state.stop_win_triggered = True
    mgr.state.day_key = 54321

    mgr.save_state()

    new_mgr = StateManager(file_path=file_path)
    assert new_mgr.load_state() is True
    assert new_mgr.state.initial_balance == 2000.0
    assert new_mgr.state.current_balance == 2100.0
    assert new_mgr.state.daily_stop_win_target == 100.0
    assert new_mgr.state.total_trades_today == 3
    assert new_mgr.state.stop_win_triggered is True
    assert new_mgr.state.day_key == 54321


def test_state_manager_load_non_existent(tmp_path):
    """Testa carregar estado de um arquivo que não existe."""
    file_path = tmp_path / "non_existent.json"
    mgr = StateManager(file_path=file_path)
    assert mgr.load_state() is False


def test_state_manager_default_path_and_original_reset(tmp_path):
    """Testa o construtor sem caminho explícito e o reset_daily_metrics diretamente."""
    mock_path = tmp_path / "default_session_state.json"
    with patch("src.infrastructure.state.state_manager.repo_path", return_value=mock_path):
        mgr = StateManager()
        assert mgr.state.initial_balance == 0.0
        mgr.reset_daily_metrics(1200.0, 60.0, 99999)
        assert mgr.state.initial_balance == 1200.0
