from types import SimpleNamespace

from src.application.services.market_audit_log import (
    format_direction_audit_line,
    format_execution_audit_line,
    format_indicators_audit_line,
    format_settlement_audit_line,
    pop_contract_audit,
    resolve_predicted_edge,
    store_contract_audit,
)
from src.domain.models.trade import TradeDirection, TradeStatus


def test_format_settlement_audit_line():
    line = format_settlement_audit_line(3, "WIN", 12.5, "CALL", "RDBULL", 0.1234)
    assert line == "[C0003] STATUS: WIN || P&L: $+12.50 || CALL || sym=RDBULL || edge=0.1234"


def test_format_direction_audit_line_with_flip():
    line = format_direction_audit_line(2, "PUT", "RDBEAR", -0.05, dl_direction="CALL")
    assert "ord=PUT" in line
    assert "dl=CALL inv" in line
    assert "edge=-0.0500" in line


def test_format_execution_audit_line():
    line = format_execution_audit_line(1, "RDBULL", "CALL", 0.72, 0.14, z_edge=0.82)
    assert line == "[C0001] EXEC_SEL | RDBULL | ord=CALL | TCN=0.72 | edge=0.1400 | Z=+0.82"


def test_format_indicators_audit_line_ignores_none_values():
    metrics = {"indicators": {"rsi": None, "hurst": 0.61}, "raw_prob": 0.32}
    line = format_indicators_audit_line(5, "RDBULL", metrics)
    assert "hurst=0.6100" in line
    assert "rsi=" not in line


def test_format_indicators_audit_line():
    metrics = {
        "indicators": {"rsi": 55.2, "hurst": 0.61},
        "raw_prob": 0.32,
        "calibrated_prob": 0.32,
        "dl_direction": "PUT",
        "exec_direction": "PUT",
    }
    line = format_indicators_audit_line(4, "RDBEAR", metrics)
    assert "[C0004] IND | RDBEAR |" in line
    assert "rsi=55.2000" in line
    assert "dl=PUT" in line


def test_format_indicators_audit_line_skips_invalid_values():
    metrics = {
        "indicators": {"rsi": "bad", "hurst": 0.61},
        "raw_prob": 0.32,
        "dl_direction": "PUT",
        "exec_direction": "PUT",
    }
    line = format_indicators_audit_line(4, "RDBEAR", metrics)
    assert "hurst=0.6100" in line
    assert "rsi=" not in line


def test_resolve_predicted_edge_prefers_payoff_key():
    assert resolve_predicted_edge({"predicted_payoff_edge": 0.42}) == 0.42


def test_store_and_pop_contract_audit():
    orch = SimpleNamespace()
    store_contract_audit(orch, 9, symbol="RDBULL", direction="CALL", edge=0.11)
    sym, direction, edge = pop_contract_audit(orch, 9)
    assert sym == "RDBULL"
    assert direction == "CALL"
    assert edge == 0.11


def test_pop_contract_audit_falls_back_to_contract_direction():
    orch = SimpleNamespace()
    contract = SimpleNamespace(direction=TradeDirection.PUT, status=TradeStatus.OPEN)
    sym, direction, edge = pop_contract_audit(orch, 4, contract=contract, symbol="RDBEAR")
    assert sym == "RDBEAR"
    assert direction == "PUT"
    assert edge == 0.0
