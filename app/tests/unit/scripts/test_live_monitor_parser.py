"""Testes do parser de logs do live monitor."""

from scripts.monitor.live_monitor import LogParser
from scripts.monitor.monitor_state import DashboardState


def test_log_parser_cluster():
    state = DashboardState()
    parser = LogParser(state)
    line = "[CLUSTER] M5 || RDBEAR: PUT (Prob: 0.377 Cal: 0.365 Edge: +0.950) || RDBULL: CALL (NEUTRO_VETO)"
    parser.process_line(line)
    assert state.last_telemetry["symbol"] == "RDBEAR"
    assert state.last_telemetry["dir"] == "PUT"
    assert state.last_telemetry["dl_dir"] == "PUT"
    assert state.last_telemetry["conv"] == "0.95"


def test_log_parser_exec():
    state = DashboardState()
    parser = LogParser(state)
    line = (
        "[C0006] EXEC || PUT [RDBEAR] || "
        "STAKE: 2.06 (DAL_L1) | PEND: 1.62 | BANCA: 87.69 || "
        "CID: 1129497159 | PAY: 1.79"
    )
    parser.process_line(line)
    assert state.last_telemetry["symbol"] == "RDBEAR"
    assert state.last_telemetry["dir"] == "PUT"
    assert state.last_telemetry["dl_dir"] == "PUT"
    assert state.last_telemetry["conv"] == "2.06"
    assert "mode=DAL_L1" in state.last_telemetry["metrics"]
    assert state.balance == 87.69


def test_log_parser_ind():
    state = DashboardState()
    parser = LogParser(state)
    line = (
        "[C0006] IND || RSI:  0.4859 | ADX:  0.2017 | HURST:  0.5671 || "
        "ATR:  -0.9558 | BBW:  -0.2226 | VOL_R:  1.0720 || Z:  +0.60 | ACC: 0.6433"
    )
    parser.process_line(line)
    assert "Z=+0.60" in state.last_telemetry["metrics"]
    assert "ACC=0.6433" in state.last_telemetry["metrics"]


def test_log_parser_session_bootstrap():
    state = DashboardState()
    parser = LogParser(state)
    parser.process_line("SESSAO INICIADA | Alvo de 2.60%: $236.62 | Stop Loss: DESATIVADO")
    assert state.session_target_win == 236.62


def test_log_parser_session_bootstrap_micro_fixed():
    state = DashboardState()
    parser = LogParser(state)
    parser.process_line("SESSAO INICIADA | Alvo fixo micro-banca: $10.00 | Stop Loss: DESATIVADO")
    assert state.session_target_win == 10.0
