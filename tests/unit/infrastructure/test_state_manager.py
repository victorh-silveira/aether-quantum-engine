from unittest.mock import patch

import pytest

from src.infrastructure.state.state_manager import SessionState, StateManager


def test_session_state_properties():
    state = SessionState()
    assert state.initial_session_balance == 0.0
    assert state.session_start_balance == 0.0
    assert state.session_profit == 0.0

    state.session_start_balance = 1000.0
    state.real_current_balance = 1050.0
    assert state.initial_balance == 1000.0
    assert state.session_profit == 50.0


def test_state_manager_init_and_reset(tmp_path):
    file_path = tmp_path / "test_session_state.json"
    mgr = StateManager(file_path=file_path)
    mgr.reset_session_metrics(1500.0, 50.0)
    assert mgr.state.initial_balance == 1500.0
    assert mgr.state.daily_stop_win_target == 50.0
    assert mgr.state.total_trades_today == 0
    assert mgr.state.stop_win_triggered is False


def test_state_manager_check_session_limits(tmp_path):
    file_path = tmp_path / "test_session_state.json"
    mgr = StateManager(file_path=file_path)
    mgr.state.daily_stop_win_target = 50.0
    mgr.state.initial_balance = 1000.0
    mgr.state.current_balance = 1060.0
    mgr.state.total_trades_today = 1
    assert mgr.check_session_limits() is True
    mgr.state.current_balance = 1040.0
    assert mgr.check_session_limits() is False


def test_state_manager_save_load_cycle(tmp_path):
    file_path = tmp_path / "test_session_state.json"
    mgr = StateManager(file_path=file_path)
    mgr.state.initial_balance = 2000.0
    mgr.state.current_balance = 2100.0
    mgr.state.daily_stop_win_target = 100.0
    mgr.state.total_trades_today = 3
    mgr.state.stop_win_triggered = True
    mgr.save_state()
    new_mgr = StateManager(file_path=file_path)
    assert new_mgr.load_state() is True
    assert new_mgr.state.daily_stop_win_target == 100.0
    assert new_mgr.state.stop_win_triggered is True


def test_state_manager_legacy_aliases(tmp_path):
    file_path = tmp_path / "legacy.json"
    mgr = StateManager(file_path=file_path)
    state = mgr.state
    state.session_start_balance = 800.0
    state.real_current_balance = 820.0
    assert state.initial_balance == 800.0
    assert state.real_current_balance == 820.0
    assert state.current_balance == 820.0
    state.initial_session_balance = 700.0
    assert state.initial_balance == 700.0
    mgr.reset_session_metrics(900.0, 9.0)
    assert mgr.state.daily_stop_win_target == 9.0


def test_state_manager_load_non_existent(tmp_path):
    mgr = StateManager(file_path=tmp_path / "non_existent.json")
    assert mgr.load_state() is False


def test_state_manager_default_path_and_original_reset(tmp_path):
    mock_path = tmp_path / "default_session_state.json"
    with patch("src.infrastructure.state.state_manager.repo_path", return_value=mock_path):
        mgr = StateManager()
        mgr.reset_session_metrics(1200.0, 60.0)
        assert mgr.state.initial_balance == 1200.0
        assert mgr.read_cached_balance() == pytest.approx(1200.0)


@pytest.mark.asyncio
async def test_state_manager_atomic_state_context_exposes_lock(tmp_path):
    mgr = StateManager(file_path=tmp_path / "lock.json")
    async with mgr.atomic_state_context():
        assert mgr._state_lock.locked()
    assert not mgr._state_lock.locked()
