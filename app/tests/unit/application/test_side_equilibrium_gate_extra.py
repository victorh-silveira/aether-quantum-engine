from unittest.mock import patch

from src.application.services.side_equilibrium_gate import (
    _log_side_eq_flip,
    log_side_equilibrium,
    resolve_direction_with_side_equilibrium,
)
from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome
from src.domain.analytics.side_equilibrium import ACTION_PASS, SideEquilibriumDecision
from src.domain.models.trade import TradeDirection


def _orch_with_side_eq(**overrides):
    cfg = {
        "orchestrator": {
            "execution": {
                "side_equilibrium": {
                    "enabled": True,
                    "small_window": 12,
                    "large_window": 100,
                    "n_min_small": 2,
                    "n_min_large": 40,
                    "wr_floor_small": 0.40,
                    "wr_floor_large": 0.48,
                    "freq_bias_max_small": 0.70,
                    "freq_bias_max_large": 0.65,
                    "kelly_mult_soft": 0.55,
                    "margin_boost_soft": 0.03,
                    **overrides,
                }
            }
        }
    }
    return type("O", (), {"config": cfg, "_side_equilibrium_hist": {}})()


def test_side_eq_rejects_flip_against_price_zone():
    orch = _orch_with_side_eq(n_min_small=2, wr_floor_small=0.40, freq_bias_max_small=0.70)
    for _ in range(4):
        record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=True)
    for _ in range(4):
        record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=False)
    for _ in range(2):
        record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=True)
    metrics = {"price_zone_direction": "CALL"}
    chosen = resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.CALL, metrics)
    assert chosen is None
    assert metrics.get("side_eq_flip_zone_conflict") is True
    assert metrics.get("gate_reason") == "side_imbalance_flip_zone_conflict"


def test_side_eq_toxic_escape_allows_flip_against_zone():
    orch = _orch_with_side_eq(n_min_small=2, wr_floor_small=0.40, freq_bias_max_small=0.70)
    for _ in range(2):
        record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=False)
    for _ in range(2):
        record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=True)
    metrics = {"price_zone_direction": "PUT"}
    chosen = resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.PUT, metrics)
    assert chosen == TradeDirection.CALL
    assert metrics.get("side_eq_flipped") is True
    assert metrics.get("side_eq_toxic_zone_escape") is True
    assert metrics.get("side_eq_toxic_escape") is True


def test_side_eq_flip_logs_once_when_resolve_called_twice():
    orch = _orch_with_side_eq(n_min_small=2, wr_floor_small=0.40, freq_bias_max_small=0.70)
    orch._active_cycle_id = 7
    for _ in range(2):
        record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=False)
    for _ in range(2):
        record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=True)
    metrics: dict = {}
    with patch("src.application.services.side_equilibrium_gate.logger") as mock_logger:
        first = resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.PUT, metrics)
        second = resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.PUT, metrics)
    assert first == TradeDirection.CALL
    assert second == TradeDirection.CALL
    flip_calls = [c for c in mock_logger.info.call_args_list if c.args and str(c.args[0]).startswith("SIDE_EQ_FLIP")]
    assert len(flip_calls) == 1


def test_side_eq_log_once_per_cycle_even_with_different_sides():
    orch = _orch_with_side_eq()
    orch._active_cycle_id = 42
    decision = SideEquilibriumDecision(
        action=ACTION_PASS,
        reason="ok",
        call_n=0,
        call_wins=0,
        put_n=0,
        put_wins=0,
        freq_bias=0.5,
        side_wr=None,
    )
    with patch("src.application.services.side_equilibrium_gate.logger") as mock_logger:
        log_side_equilibrium(decision, symbol="R_10", proposed=TradeDirection.PUT, orch=orch)
        log_side_equilibrium(decision, symbol="R_10", proposed=TradeDirection.CALL, orch=orch)
        log_side_equilibrium(decision, symbol="R_10", proposed=TradeDirection.PUT, orch=None)
    assert mock_logger.info.call_count == 1


def test_side_eq_both_sides_hard_skip_returns_none_and_blocks_replay():
    orch = _orch_with_side_eq(n_min_small=2, wr_floor_small=0.40, freq_bias_max_small=0.70)
    for _ in range(2):
        record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=False)
        record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=False)
    metrics: dict = {}
    assert resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.PUT, metrics) is None
    assert metrics.get("side_eq_blocked") is True
    assert resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.PUT, metrics) is None


def test_side_eq_pass_and_sticky_blocked_gate_reasons():
    orch = _orch_with_side_eq()
    passed = {"gate_reason": "side_imbalance_small_n", "quality_guard_reject": True}
    assert resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.CALL, passed) == TradeDirection.CALL
    assert passed.get("gate_reason") is None and passed.get("side_eq_action") == "pass"
    blocked = {"side_eq_gate_done": True, "side_eq_blocked": True, "side_eq_reason": "side_imbalance_small_n"}
    assert resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.CALL, blocked) is None
    assert blocked["gate_reason"] == "side_imbalance_small_n" and blocked["quality_guard_reject"] is True


def test_side_eq_gate_done_invalid_direction_name_falls_back():
    orch = _orch_with_side_eq()
    metrics = {"side_eq_gate_done": True, "side_eq_blocked": False, "exec_direction": "HOLD"}
    assert resolve_direction_with_side_equilibrium(orch, "R_10", TradeDirection.CALL, metrics) == TradeDirection.CALL


def test_side_eq_flip_log_creates_bag_and_dedupes_without_gate_flag():
    orch = _orch_with_side_eq(n_min_small=2, wr_floor_small=0.40, freq_bias_max_small=0.70)
    orch._active_cycle_id = 9
    orch._side_eq_log_keys = "not-a-set"
    with patch("src.application.services.side_equilibrium_gate.logger") as mock_logger:
        _log_side_eq_flip(
            orch,
            symbol="R_10",
            proposed=TradeDirection.PUT,
            opposite=TradeDirection.CALL,
            reason="side_imbalance_small_n",
        )
        _log_side_eq_flip(
            orch,
            symbol="R_10",
            proposed=TradeDirection.PUT,
            opposite=TradeDirection.CALL,
            reason="side_imbalance_small_n",
        )
    assert mock_logger.info.call_count == 1
    assert isinstance(orch._side_eq_log_keys, set)
