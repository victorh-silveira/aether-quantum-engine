from src.application.services.execution_micro_follow import (
    apply_micro_discord_follow_candle,
    candle_follow_edge_ok,
)


def test_candle_follow_edge_ok_and_blocked():
    metrics = {"calibrated_prob": 0.56, "min_edge_explore": 0.015}
    assert candle_follow_edge_ok(metrics, "CALL", cfg={}) is True
    assert "micro_discord_follow_blocked" not in metrics
    weak = {"calibrated_prob": 0.50}
    assert candle_follow_edge_ok(weak, "CALL", cfg={"min_edge_recovery": 0.015}) is False
    assert weak["micro_discord_follow_blocked"] == "edge_nonpos"
    sub = {"calibrated_prob": 0.545}
    assert candle_follow_edge_ok(sub, "CALL", cfg={}) is False
    assert sub["micro_discord_follow_blocked"] == "edge_subfloor"


def test_candle_follow_min_edge_from_cfg_and_bad_values():
    metrics = {"calibrated_prob": 0.56}
    assert candle_follow_edge_ok(metrics, "CALL", cfg={"min_edge_explore": "x"}) is True
    assert float(metrics["micro_discord_follow_min_edge"]) == 0.015
    metrics2 = {"calibrated_prob": 0.56, "min_edge_explore": -1.0}
    assert candle_follow_edge_ok(metrics2, "CALL", cfg={"min_edge_recovery": 0.02}) is True
    assert float(metrics2["micro_discord_follow_min_edge"]) == 0.02
    metrics3 = {"calibrated_prob": 0.56}
    assert candle_follow_edge_ok(metrics3, "CALL", cfg=None) is True
    assert float(metrics3["micro_discord_follow_min_edge"]) == 0.015


def test_apply_follow_noop_when_disabled_or_edge_fail():
    metrics = {
        "exec_direction": "PUT",
        "calibrated_prob": 0.56,
        "kelly_fraction_scale": 1.0,
    }
    assert (
        apply_micro_discord_follow_candle(
            metrics,
            candle="CALL",
            exec_side="PUT",
            body=2.0,
            cfg={"micro_discord_follow_candle": False},
        )
        is False
    )
    assert (
        apply_micro_discord_follow_candle(
            metrics,
            candle="CALL",
            exec_side="PUT",
            body=2.0,
            cfg={"micro_discord_follow_candle": True, "min_edge_explore": 0.20},
        )
        is False
    )


def test_apply_follow_flips_and_soft():
    metrics = {
        "exec_direction": "PUT",
        "resolved_direction": "PUT",
        "calibrated_prob": 0.56,
        "kelly_fraction_scale": 1.0,
        "fusion_applied": True,
        "fusion_p_call": 0.58,
        "fusion_p_put": 0.42,
    }
    assert (
        apply_micro_discord_follow_candle(
            metrics,
            candle="CALL",
            exec_side="PUT",
            body=3.0,
            cfg={
                "micro_discord_follow_candle": True,
                "micro_discord_follow_kelly_mult": 0.55,
                "min_edge_explore": 0.015,
            },
        )
        is True
    )
    assert metrics["exec_direction"] == "CALL"
    assert metrics["fusion_p_eff"] == 0.58
    assert metrics["micro_discord_follow_soft"] is True
    assert metrics["kelly_fraction_scale"] == 0.55
