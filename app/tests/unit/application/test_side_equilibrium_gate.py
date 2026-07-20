from unittest.mock import MagicMock

from src.application.services.side_equilibrium_gate import (
    apply_side_equilibrium_to_metrics,
    evaluate_proposed_side_equilibrium,
)
from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome
from src.domain.analytics.side_equilibrium import ACTION_PASS
from src.domain.models.trade import TradeDirection


def _orch_with_side_eq(**overrides):
    cfg = {
        "orchestrator": {
            "execution": {
                "side_equilibrium": {
                    "enabled": True,
                    "small_window": 12,
                    "large_window": 100,
                    "n_min_small": 6,
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


def test_record_and_hard_skip_call_bias_regression():
    orch = _orch_with_side_eq()
    for _ in range(8):
        record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=False)
    for _ in range(2):
        record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=True)
    decision = evaluate_proposed_side_equilibrium(orch, "R_10", TradeDirection.CALL)
    assert decision.action == "hard_skip"
    metrics = {"kelly_fraction_scale": 1.0, "quality_min_direction_margin": 0.02}
    blocked = apply_side_equilibrium_to_metrics(metrics, decision, proposed=TradeDirection.CALL)
    assert blocked is True
    assert metrics["gate_reason"].startswith("side_imbalance_small_n")
    assert metrics.get("exec_direction") is None


def test_soft_penalty_sets_kelly_and_margin_without_flip():
    orch = _orch_with_side_eq(n_min_small=100)
    for i in range(50):
        record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=(i % 5 == 0))
    for _ in range(50):
        record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=True)
    decision = evaluate_proposed_side_equilibrium(orch, "R_10", TradeDirection.CALL)
    assert decision.action == "soft_penalty"
    metrics = {"kelly_fraction_scale": 1.0, "quality_min_direction_margin": 0.02}
    blocked = apply_side_equilibrium_to_metrics(metrics, decision, proposed=TradeDirection.CALL)
    assert blocked is False
    assert metrics["kelly_fraction_scale"] == 0.55
    assert metrics["quality_min_direction_margin"] == 0.05


def test_evaluate_proposed_disabled_and_no_context():
    assert evaluate_proposed_side_equilibrium(None, "R_10", TradeDirection.CALL).reason == "no_context"
    orch = _orch_with_side_eq(enabled=False)
    assert evaluate_proposed_side_equilibrium(orch, "R_10", TradeDirection.CALL).reason == "disabled"
    assert evaluate_proposed_side_equilibrium(orch, None, TradeDirection.CALL).action == ACTION_PASS


def test_record_ignores_invalid_direction_and_persists_best_effort():
    orch = _orch_with_side_eq()
    counts = record_side_equilibrium_outcome(orch, "R_10", direction="HOLD", won=True)
    assert counts.call_n == 0
    client = MagicMock()
    orch.state_store = MagicMock(client=client)
    writer = MagicMock()
    writer.enqueue_trade_outcome = MagicMock(return_value=None)
    orch.timescale_writer = writer
    record_side_equilibrium_outcome(
        orch,
        "R_10",
        direction="CALL",
        won=True,
        profit=1.2,
        raw_prob=0.61,
        calibrated_prob=0.62,
        cycle_id=9,
    )
    client.hset.assert_called()
    writer.enqueue_trade_outcome.assert_called()


def test_persist_redis_and_timescale_exception_paths():
    orch = _orch_with_side_eq()
    client = MagicMock()
    client.hset.side_effect = RuntimeError("redis down")
    orch.state_store = MagicMock(client=client)
    writer = MagicMock()
    writer.enqueue_trade_outcome.side_effect = RuntimeError("db down")
    orch.timescale_writer = writer
    counts = record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=False)
    assert counts.put_n == 1
    bare = type("C", (), {})()
    orch.state_store = MagicMock(client=bare)
    record_side_equilibrium_outcome(orch, "R_10", direction="PUT", won=True)

    class _Awaitable:
        def __await__(self):
            if False:
                yield None

    writer.enqueue_trade_outcome.side_effect = None
    writer.enqueue_trade_outcome.return_value = _Awaitable()
    orch.create_task = MagicMock()
    record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=True)
    orch.create_task.assert_called()
    client_ok = MagicMock()
    client_ok.hset.return_value = _Awaitable()
    orch.state_store = MagicMock(client=client_ok)
    orch.timescale_writer = None
    orch.infra = MagicMock(timescale_writer=None)
    record_side_equilibrium_outcome(orch, "R_10", direction="CALL", won=False)
