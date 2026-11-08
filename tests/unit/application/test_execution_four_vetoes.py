"""Contrato simetrico dos quatro vetos e da inversao com EV verificavel."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.application.services import execution_four_vetoes as policy
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_senior_skips import (
    apply_senior_execution_skips,
    should_skip_tcn_noise_discord,
)
from src.application.services.market_audit_gate_tokens import format_gates_audit_line
from src.application.services.orchestrator.execution_blockers import _candidate_block_reason
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection as Side


CONFIG = {"four_market_vetoes": True, "market_direction_trigger": True, "min_edge_execute": 0.01}


def setup_metrics(side, kind="rejection"):
    """Retorna extremos espelhados sem candles futuros."""
    if kind == "rejection":
        return {
            "closed_candle_ohlc": (100, 110, 99, 101) if side == Side.CALL else (100, 101, 90, 99),
            "indicators": {"rsi": 0.8 if side == Side.CALL else 0.2, "bb_pct_b": 1.1 if side == Side.CALL else -0.1},
        }
    opposite = "PUT" if side == Side.CALL else "CALL"
    return {
        "closed_candle_ohlc": (109, 110, 99, 100) if side == Side.CALL else (100, 110, 99, 109),
        "indicators": {"di_diff": -0.3 if side == Side.CALL else 0.3},
        "trend_direction": opposite,
        "previous_closed_candle_direction": opposite,
    }


@pytest.mark.parametrize(
    "side,kind,reason",
    [
        (Side.CALL, "rejection", "call_top_rejection"),
        (Side.PUT, "rejection", "put_bottom_rejection"),
        (Side.CALL, "continuation", "call_down_continuation"),
        (Side.PUT, "continuation", "put_up_continuation"),
    ],
)
def test_exact_four_market_reasons(side, kind, reason):
    metrics = setup_metrics(side, kind)
    assert policy.market_veto_reason(side, metrics) == reason
    assert policy.market_veto_reason(Side.PUT if side == Side.CALL else Side.CALL, metrics) is None
    assert apply_senior_execution_skips(side, metrics, exec_cfg=CONFIG) == (side, True)
    assert _candidate_block_reason(metrics) == reason
    assert metrics["execution_candidate_ready"] is False


@pytest.mark.parametrize(
    "candle", [None, (100, 99, 90, 101), (100, 100, 100, 100), (100, float("nan"), 90, 101), (0, 2, 0, 1)]
)
def test_invalid_candle_never_creates_setup(candle):
    assert policy.market_veto_reason(Side.CALL, {"closed_candle_ohlc": candle}) is None


def test_missing_or_invalid_indicators_and_ordinary_trend_do_not_veto():
    metrics = {"closed_candle_ohlc": (100, 110, 99, 101), "rsi": "bad", "bb_pct_b": float("inf")}
    assert policy.market_veto_reason(Side.CALL, metrics) is None
    metrics.update(indicators={"rsi": 0.5, "bb_pct_b": 0.5}, trend_direction="PUT")
    assert apply_senior_execution_skips(Side.CALL, metrics, exec_cfg={**CONFIG, "skip_trend_discord": True}) == (
        Side.CALL,
        False,
    )


@pytest.mark.parametrize("config", [None, {}, {"four_market_vetoes": "true"}, {"four_market_vetoes": True}])
def test_trigger_requires_explicit_boolean_activation(config):
    metrics = setup_metrics(Side.CALL)
    assert policy.reevaluate_market_direction(Side.CALL, metrics, config, payout=0.85) == Side.CALL
    assert metrics["market_trigger_applied"] is False


def test_no_setup_records_status():
    metrics = {}
    assert policy.reevaluate_market_direction(Side.CALL, metrics, CONFIG, payout=0.85) == Side.CALL
    assert metrics["market_trigger_status"] == "no_extreme_setup"


def test_trigger_clears_previous_candidate_when_setup_disappears():
    metrics = {
        "market_trigger_status": "candidate_without_edge",
        "market_trigger_candidate": "PUT",
        "market_trigger_setup": "call_down_continuation",
        "market_trigger_candidate_edge": -0.1,
    }
    assert policy.reevaluate_market_direction(Side.CALL, metrics, CONFIG, payout=0.85) == Side.CALL
    assert metrics["market_trigger_status"] == "no_extreme_setup"
    assert "market_trigger_candidate" not in metrics
    assert "market_trigger_setup" not in metrics
    assert "market_trigger_candidate_edge" not in metrics


def test_trigger_respects_quote_safety_margin_before_switching_side():
    metrics = {**setup_metrics(Side.PUT), "calibrated_prob": 0.57}
    config = {**CONFIG, "quote_safety_margin": 0.01}
    assert policy.reevaluate_market_direction(Side.PUT, metrics, config, payout=0.784) == Side.PUT
    assert metrics["market_trigger_status"] == "candidate_below_quote_margin"
    assert metrics["market_trigger_candidate_edge"] > config["min_edge_execute"]


def test_market_trigger_cancels_prior_flip_flags_when_model_supports_candidate():
    metrics = {
        **setup_metrics(Side.PUT),
        "calibrated_prob": 0.7,
        "loss_clf_flip": True,
        "anti_trend_lock_flip": True,
        "alpha_flip_applied": True,
    }
    assert policy.reevaluate_market_direction(Side.PUT, metrics, CONFIG, payout=0.784) == Side.CALL
    assert metrics["market_trigger_applied"] is True
    assert metrics["loss_clf_flip"] is False
    assert metrics["anti_trend_lock_flip"] is False
    assert metrics["alpha_flip_applied"] is False


@pytest.mark.parametrize(
    "prob,payout,floor",
    [
        (None, 0.85, 0.01),
        (-0.1, 0.85, 0.01),
        (1.1, 0.85, 0.01),
        (0.3, float("nan"), 0.01),
        (0.3, 0, 0.01),
        (0.3, 0.85, "bad"),
        (0.3, 0.85, -1),
    ],
)
def test_trigger_rejects_invalid_probability_payout_or_floor(prob, payout, floor):
    metrics = {**setup_metrics(Side.CALL), "calibrated_prob": prob}
    assert (
        policy.reevaluate_market_direction(Side.CALL, metrics, {**CONFIG, "min_edge_execute": floor}, payout=payout)
        == Side.CALL
    )
    assert metrics["market_trigger_status"] == "invalid_probability_or_payout"


@pytest.mark.parametrize("side,prob", [(Side.CALL, 0.52), (Side.PUT, 0.48)])
def test_negative_edge_is_not_opposite_side_evidence(side, prob):
    metrics = {**setup_metrics(side), "calibrated_prob": prob, "pending_loss_total": 500}
    assert policy.reevaluate_market_direction(side, metrics, CONFIG, payout=0.784) == side
    assert metrics["market_trigger_status"] == "candidate_without_edge"
    assert metrics["market_trigger_candidate_edge"] < 0


@pytest.mark.parametrize("side,prob,candidate", [(Side.CALL, 0.3, Side.PUT), (Side.PUT, 0.7, Side.CALL)])
def test_trigger_uses_probability_of_new_side(side, prob, candidate):
    metrics = {**setup_metrics(side), "calibrated_prob": prob, "loss_clf_flip": True}
    assert policy.reevaluate_market_direction(side, metrics, CONFIG, payout=0.784) == candidate
    assert metrics["cal_side_edge"] == pytest.approx(0.7 * 1.784 - 1)
    assert metrics["conviction"] == pytest.approx(0.7)
    assert metrics["resolved_direction"] == candidate.name
    assert metrics["market_trigger_applied"] is True
    assert metrics["loss_clf_flip"] is False


def test_candidate_is_checked_for_its_own_extreme():
    metrics = setup_metrics(Side.CALL)
    metrics.update(calibrated_prob=0.3, trend_direction="CALL", previous_closed_candle_direction="CALL")
    metrics["closed_candle_ohlc"] = (100, 110, 99, 109)
    metrics["indicators"]["di_diff"] = 0.3
    metrics["indicators"].update(rsi=0.8, bb_pct_b=1.1)
    assert policy.market_veto_reason(Side.PUT, metrics) == "put_up_continuation"


def test_conflicting_candidate_is_not_accepted(monkeypatch):
    monkeypatch.setattr(policy, "market_veto_reason", lambda *args, **kwargs: "extreme")
    metrics = {"calibrated_prob": 0.3}
    assert policy.reevaluate_market_direction(Side.CALL, metrics, CONFIG, payout=0.85) == Side.CALL
    assert metrics["market_trigger_status"] == "candidate_vetoed"


def test_four_veto_policy_preserves_economic_gate():
    metrics = {"cal_side_edge": -0.03}
    assert apply_senior_execution_skips(Side.CALL, metrics, exec_cfg={**CONFIG, "skip_neg_edge": True}) == (
        Side.CALL,
        True,
    )
    assert metrics["gate_reason"] == "neg_edge"


def test_trigger_audit_is_visible():
    line = format_gates_audit_line(
        {"market_trigger_status": "candidate_without_edge", "market_trigger_candidate": "PUT"}
    )
    assert "trigger=candidate_without_edge candidate=PUT" in line


def test_resolver_rechecks_direction_before_economic_gate():
    metrics = {**setup_metrics(Side.CALL), "calibrated_prob": 0.3, "raw_prob": 0.3, "deploy_ok": True}
    result = resolve_execution_direction(
        {"metrics": metrics}, exec_cfg={**CONFIG, "invert_exec_side": True, "skip_neg_edge": True}
    )
    assert result is not None
    direction, resolved = result
    assert direction == Side.PUT
    assert resolved["direction_origin"] == "MARKET_TRIGGER_TCN"
    assert resolved["market_trigger_applied"] is True
    assert resolved["cal_side_edge"] > 0.01
    assert resolved["kelly_side_p"] == pytest.approx(0.7)


def test_resolver_never_inverts_weak_tcn_call_from_negative_edge():
    metrics = {**setup_metrics(Side.CALL), "calibrated_prob": 0.52, "raw_prob": 0.52, "deploy_ok": True}
    entry = {"metrics": metrics}
    assert resolve_execution_direction(entry, exec_cfg={**CONFIG, "skip_neg_edge": True}) is None
    assert entry["metrics"]["market_trigger_status"] == "candidate_without_edge"
    assert entry["metrics"]["gate_reason"] == "neg_edge"


def test_continuation_requires_two_distinct_closed_candles():
    metrics = setup_metrics(Side.CALL, "continuation")
    metrics["scale_micro_prev_bar_dir"] = "PUT"
    current = Candle("1HZ75V", 109, 110, 99, 100, datetime.now(UTC), 600)
    previous = Candle("1HZ75V", 100, 110, 99, 109, datetime.now(UTC), 300)
    forming = Candle("1HZ75V", 100, 101, 99, 100, datetime.now(UTC), 900)
    orch = SimpleNamespace(stream=SimpleNamespace(micro_candles={"1HZ75V": [previous, current, forming]}))
    assert policy.market_veto_reason(Side.CALL, metrics, orch=orch, symbol="1HZ75V") is None
    orch.stream.micro_candles["1HZ75V"] = [current, current, forming]
    assert policy.market_veto_reason(Side.CALL, metrics, orch=orch, symbol="1HZ75V") is None
    previous_bearish = Candle("1HZ75V", 109, 110, 99, 100, datetime.now(UTC), 300)
    orch.stream.micro_candles["1HZ75V"] = [previous_bearish, current, forming]
    assert policy.market_veto_reason(Side.CALL, metrics, orch=orch, symbol="1HZ75V") == "call_down_continuation"
    orch.stream.micro_candles["1HZ75V"] = [current, forming]
    assert policy.market_veto_reason(Side.CALL, metrics, orch=orch, symbol="1HZ75V") is None


def test_real_market_oversold_veto():
    metrics = {
        "closed_candle_ohlc": (100, 101, 95, 99),
        "indicators": {"rsi": 0.293, "bb_pct_b": 0.038},
    }
    assert policy.market_veto_reason(Side.PUT, metrics) == "put_bottom_rejection"
    assert policy.market_veto_reason(Side.CALL, metrics) is None


def test_tcn_noise_discord_skips_and_edge_bypass():
    metrics = {
        "direction_margin": 0.005,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.03,
    }
    assert should_skip_tcn_noise_discord(metrics, Side.CALL, {"skip_tcn_noise_discord": True}) is False

    metrics_invalid = {"direction_margin": "invalid_num"}
    assert should_skip_tcn_noise_discord(metrics_invalid, Side.CALL, {"skip_tcn_noise_discord": True}) is False

    metrics_invalid_edge = {
        "direction_margin": 0.005,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": "invalid_edge",
    }
    assert should_skip_tcn_noise_discord(metrics_invalid_edge, Side.CALL, {"skip_tcn_noise_discord": True}) is True

    metrics_block = {
        "direction_margin": 0.005,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.01,
    }
    res_dir, blocked = apply_senior_execution_skips(
        Side.CALL,
        metrics_block,
        exec_cfg={"four_market_vetoes": True, "skip_tcn_noise_discord": True},
    )
    assert blocked is True
    assert res_dir == Side.CALL
    assert metrics_block["gate_reason"] == "tcn_noise_discord"


def test_exhaustion_turn_oversold_candle_reversal():
    """Verifica que sobre-extensao com vela de reversao fecha o setup de rejeicao."""
    metrics_put = {
        "closed_candle_ohlc": (100, 105, 99, 104),
        "indicators": {"rsi": 0.25, "bb_pct_b": 0.10},
    }
    assert policy.market_veto_reason(Side.PUT, metrics_put) == "put_bottom_rejection"
    assert policy.market_veto_reason(Side.CALL, metrics_put) is None

    metrics_call = {
        "closed_candle_ohlc": (104, 105, 99, 100),
        "indicators": {"rsi": 0.75, "bb_pct_b": 0.90},
    }
    assert policy.market_veto_reason(Side.CALL, metrics_call) == "call_top_rejection"
    assert policy.market_veto_reason(Side.PUT, metrics_call) is None


def test_four_vetoes_honors_counter_trend_with_opposing_candle():
    """Verifica bloqueio de contra-tendencia com candle discordante sob four_market_vetoes."""
    metrics = {
        "trend_direction": "PUT",
        "closed_micro_candle_stamped": True,
        "closed_micro_candle_dir": "PUT",
        "cal_side_edge": 0.02,
    }
    res_dir, blocked = apply_senior_execution_skips(
        Side.CALL,
        metrics,
        exec_cfg={"four_market_vetoes": True, "skip_trend_discord": True, "counter_trend_min_edge": 0.08},
    )
    assert blocked is True
    assert res_dir == Side.CALL
    assert metrics["gate_reason"] == "counter_trend_unconfirmed"


def test_calibrated_rejection_at_realistic_thresholds():
    metrics_call = {
        "closed_candle_ohlc": (100, 106, 99, 103),
        "indicators": {"rsi": 0.69, "bb_pct_b": 0.89},
    }
    assert policy.market_veto_reason(Side.CALL, metrics_call) == "call_top_rejection"

    metrics_put = {
        "closed_candle_ohlc": (100, 101, 94, 97),
        "indicators": {"rsi": 0.31, "bb_pct_b": 0.11},
    }
    assert policy.market_veto_reason(Side.PUT, metrics_put) == "put_bottom_rejection"
