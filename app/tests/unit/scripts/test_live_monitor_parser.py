"""Testes do parser de logs do live monitor."""

from scripts.monitor.live_monitor import LogParser
from scripts.monitor.monitor_state import DashboardState


def test_log_parser_cluster():
    state = DashboardState()
    parser = LogParser(state)
    line = (
        "[CLUSTER] M5 || R_10: PUT (Prob: 0.377 Cal: 0.365 Margin: 0.135 Edge: +0.950) || "
        "R_10: CALL (Prob: 0.500 Cal: 0.500 Margin: 0.000 Edge: +0.000 | NEUTRO_VETO)"
    )
    parser.process_line(line)
    assert state.last_telemetry["symbol"] == "R_10"
    assert state.last_telemetry["dir"] == "PUT"
    assert state.last_telemetry["dl_dir"] == "PUT"
    assert state.last_telemetry["conv"] == "0.95"


def test_log_parser_exec():
    state = DashboardState()
    parser = LogParser(state)
    for line in (
        "[C0006] EXEC || PUT [R_10] || STAKE: 2.00 (RECOVER_DAL_L1)",
        "[C0006] EXEC || PEND: 1.00 | LIN: 1 | CAP: 50.00 | BANCA: 87.69",
        "[C0006] EXEC || CID: 1129497159 | PAY: 1.79",
    ):
        parser.process_line(line)
    assert state.last_telemetry["symbol"] == "R_10"
    assert state.last_telemetry["dir"] == "PUT"
    assert state.last_telemetry["dl_dir"] == "PUT"
    assert state.last_telemetry["conv"] == "2.00"
    assert "mode=RECOVER_DAL_L1" in state.last_telemetry["metrics"]
    assert "lin=1" in state.last_telemetry["metrics"]
    assert "pay=1.79" in state.last_telemetry["metrics"]
    assert state.balance == 87.69


def test_log_parser_exec_legacy_recover_dal():
    state = DashboardState()
    parser = LogParser(state)
    line = (
        "[C0006] EXEC || PUT [R_10] || "
        "STAKE: 2.06 (RECOVER_DAL_L1) | PEND: 1.62 | LIN: 1 | CAP: 4.20 | "
        "BANCA: 87.69 || "
        "CID: 1129497159 | PAY: 1.79"
    )
    parser.process_line(line)
    assert "mode=RECOVER_DAL_L1" in state.last_telemetry["metrics"]


def test_log_parser_exec_legacy_without_lin_cap():
    state = DashboardState()
    parser = LogParser(state)
    line = (
        "[C0006] EXEC || PUT [R_10] || STAKE: 2.06 (DAL_L1) | PEND: 1.62 | BANCA: 87.69 || CID: 1129497159 | PAY: 1.79"
    )
    parser.process_line(line)
    assert "mode=DAL_L1" in state.last_telemetry["metrics"]
    assert state.balance == 87.69


def test_log_parser_ind():
    state = DashboardState()
    parser = LogParser(state)
    for line in (
        "[C0006] IND || RSI:  0.4859 | ADX:  0.2017 | HURST:  0.5671",
        "[C0006] IND || ATR:  -0.9558 | BBW:  -0.2226 | VOL_R:  1.0720",
        "[C0006] IND || Z:  +0.60 | ACC: 0.6433 | MARGIN: 0.120 | CAL_EDGE: -0.062",
        "[C0006] IND || NEUTRAL: calibrated | META_VETO: none || SCALE: tcn=PUT tape=PUT",
    ):
        parser.process_line(line)
    assert "Z=+0.60" in state.last_telemetry["metrics"]
    assert "ACC=0.6433" in state.last_telemetry["metrics"]
    assert "MARGIN=0.120" in state.last_telemetry["metrics"]
    assert "CAL_EDGE=-0.062" in state.last_telemetry["metrics"]
    assert "NEUTRAL=calibrated" in state.last_telemetry["metrics"]
    assert "META_VETO=none" in state.last_telemetry["metrics"]


def test_log_parser_ind_legacy_single_line():
    state = DashboardState()
    parser = LogParser(state)
    line = (
        "[C0006] IND || RSI:  0.4859 | ADX:  0.2017 | HURST:  0.5671 || "
        "ATR:  -0.9558 | BBW:  -0.2226 | VOL_R:  1.0720 || Z:  +0.60 | ACC: 0.6433 || "
        "MARGIN: 0.120 | NEUTRAL: calibrated | META_VETO: none"
    )
    parser.process_line(line)
    assert "Z=+0.60" in state.last_telemetry["metrics"]
    assert "NEUTRAL=calibrated" in state.last_telemetry["metrics"]


def test_log_parser_session_bootstrap():
    state = DashboardState()
    parser = LogParser(state)
    parser.process_line("SESSAO INICIADA | Alvo de 2.60%: $236.62 | Stop Loss: DESATIVADO")
    assert state.session_target_win == 236.62


def test_log_parser_session_bootstrap_micro_fixed():
    state = DashboardState()
    parser = LogParser(state)
    parser.process_line("SESSAO INICIADA | Alvo fixo micro-banca: $10.00 | Stop Loss: DESATIVADO | banca=$95.00")
    assert state.session_target_win == 10.0
