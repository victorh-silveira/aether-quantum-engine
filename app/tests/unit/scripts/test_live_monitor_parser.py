"""Testes do parser de logs do live monitor."""

from scripts.monitor.live_monitor import LogParser
from scripts.monitor.monitor_state import DashboardState


def test_log_parser_exec_sel():
    state = DashboardState()
    parser = LogParser(state)
    line = (
        "[C0001] EXEC_SEL | RDBULL ord=CALL dl=PUT s=0.72 v=0.65 r=0.58 | "
        "P(CALL)=0.58 P(PUT)=0.42 | Acc=0.62 Score=0.72 | Votes: CALL=3 PUT=2 | rsi=0.55 cmo=0.12"
    )
    parser.process_line(line)
    assert state.last_telemetry["symbol"] == "RDBULL"
    assert state.last_telemetry["dir"] == "CALL"
    assert state.last_telemetry["dl_dir"] == "PUT"
    assert state.last_telemetry["conv"] == "0.72"
    assert "rsi=0.55" in state.last_telemetry["metrics"]


def test_log_parser_session_bootstrap():
    state = DashboardState()
    parser = LogParser(state)
    parser.process_line("SESSAO INICIADA | Alvo de 1%: $91.01 | Stop Loss: DESATIVADO")
    assert state.session_target_win == 91.01
