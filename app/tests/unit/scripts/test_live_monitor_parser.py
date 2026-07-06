"""Testes do parser de logs do live monitor."""

from scripts.monitor.live_monitor import LogParser
from scripts.monitor.monitor_state import DashboardState


def test_log_parser_dir_sel():
    state = DashboardState()
    parser = LogParser(state)
    line = "[C0001] DIR_SEL || ord=CALL || dl=PUT inv || sym=RDBULL || edge=0.7200"
    parser.process_line(line)
    assert state.last_telemetry["symbol"] == "RDBULL"
    assert state.last_telemetry["dir"] == "CALL"
    assert state.last_telemetry["dl_dir"] == "PUT"
    assert state.last_telemetry["conv"] == "0.72"


def test_log_parser_exec_sel():
    state = DashboardState()
    parser = LogParser(state)
    line = "[C0001] EXEC_SEL | RDBULL | ord=CALL | TCN=0.72 | edge=0.1400 (Z=+0.82) | WIN_EXPECTED"
    parser.process_line(line)
    assert state.last_telemetry["symbol"] == "RDBULL"
    assert state.last_telemetry["dir"] == "CALL"
    assert state.last_telemetry["dl_dir"] == "CALL"
    assert state.last_telemetry["conv"] == "0.72"
    assert "Z=+0.82" in state.last_telemetry["metrics"]


def test_log_parser_session_bootstrap():
    state = DashboardState()
    parser = LogParser(state)
    parser.process_line("SESSAO INICIADA | Alvo de 1%: $91.01 | Stop Loss: DESATIVADO")
    assert state.session_target_win == 91.01
