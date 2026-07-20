from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.deep_learning.dl_indicator_prob_blend import (
    blend_prob_with_indicator_consensus,
    indicator_vote_share,
)
from src.application.services.execution_direction import (
    build_execution_candidate,
    mandatory_execution_eligible,
)
from src.application.services.execution_direction_checks import (
    _macro_indicator_float,
    _rsi_di_oppose_direction,
    apply_technical_agreement,
    initial_direction_checks,
)
from src.application.services.execution_direction_fallback import (
    _last_resort_fallback_pick,
    _scored_fallback_pick,
)
from src.application.services.execution_symbols import select_mandatory_execution_candidate
from src.domain.analytics.side_equilibrium import binomial_z_vs_p
from src.domain.models.trade import TradeDirection


def test_indicator_vote_share_empty_and_votes_low_blend():
    assert indicator_vote_share(0, 0) == (0.5, 0.5, 0)
    prob, delta, reason = blend_prob_with_indicator_consensus(0.50, 1, 1, min_votes=4)
    assert reason == "votes_low"
    assert delta == 0.0
    assert prob == 0.50


def test_mandatory_eligible_requires_inferable_direction():
    assert mandatory_execution_eligible({"metrics": {"deploy_ok": True}}) is False


def test_build_execution_candidate_reads_peer_from_decisions():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.72,
            "raw_prob": 0.72,
            "deploy_ok": True,
            "execute": True,
            "predicted_payoff_edge": 0.2,
            "edge_zscore": 1.0,
            "edge_zscore_samples": 20,
        },
    }
    peer = {
        "direction": TradeDirection.PUT,
        "metrics": {"calibrated_prob": 0.30, "raw_prob": 0.30, "deploy_ok": True},
    }
    with patch("src.application.services.execution_direction.hedge_peer", return_value="R_25"):
        built = build_execution_candidate(
            "R_10",
            entry,
            decisions={"R_10": entry, "R_25": peer},
            orch=SimpleNamespace(config={"orchestrator": {"execution": {}}}),
        )
    assert built is not None
    assert built[0] == "R_10"


def test_select_mandatory_empty_pool_returns_none():
    assert (
        select_mandatory_execution_candidate(
            None,
            [("R_10", TradeDirection.CALL, {"trade_score": 0.7})],
            last_loss_symbol="R_10",
            recovery_active=True,
            skip_symbols=frozenset({"R_10"}),
        )
        is None
    )


def test_macro_indicator_and_rsi_di_oppose_paths():
    assert _macro_indicator_float({}, "rsi") is None
    assert _macro_indicator_float({"indicators": {"rsi": "x"}}, "rsi") is None
    assert _macro_indicator_float({"macro_indicators": {"rsi": 0.62}}, "rsi") == 0.62
    assert _rsi_di_oppose_direction({"indicators": {"rsi": 0.51, "di_diff": 0.1}}, TradeDirection.PUT) is False
    assert _rsi_di_oppose_direction({"macro_indicators": {"rsi": 0.70, "di_diff": 0.2}}, TradeDirection.PUT) is True


def test_discordance_side_flag_dt_override_and_sniper_freeze():
    side_metrics = {"call_votes": 0, "put_votes": 0, "macro_indicators": {"rsi": 0.70, "di_diff": 0.2}}
    _, side_veto = apply_technical_agreement(side_metrics, TradeDirection.PUT, 0.55, {"discordance_veto_enabled": True})
    assert side_veto is True
    assert side_metrics.get("indicator_side_discordance") is True
    dt_metrics = {"call_votes": 1, "put_votes": 4, "trend_direction": "PUT"}
    _, veto = apply_technical_agreement(
        dt_metrics,
        TradeDirection.CALL,
        0.55,
        {"discordance_veto_enabled": True, "dynamic_threshold": {"require_indicator_consensus": True}},
    )
    assert veto is True
    assert dt_metrics.get("indicator_trend_discordance") is True
    disc = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.70,
            "raw_prob": 0.70,
            "deploy_ok": True,
            "execute": True,
            "call_votes": 1,
            "put_votes": 4,
            "trend_direction": "PUT",
        },
    }
    with (
        patch("src.application.services.execution_direction_checks.apply_hurst_noise_veto", return_value=False),
        patch("src.application.services.execution_direction_checks.apply_bb_squeeze_requirement", return_value=False),
    ):
        assert (
            initial_direction_checks(disc, {"discordance_veto_enabled": True, "require_indicator_consensus": True})
            is None
        )
    assert disc["metrics"].get("gate_reason") == "indicator_discordance"
    sniper = {
        "direction": TradeDirection.CALL,
        "metrics": {"calibrated_prob": 0.70, "raw_prob": 0.70, "deploy_ok": True, "execute": True},
    }
    with patch("src.application.services.execution_direction_checks.apply_hurst_noise_veto", return_value=True):
        assert initial_direction_checks(sniper, {}) is None


def test_fallback_picks_cover_skip_score_and_success_return():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "trade_score": 0.80,
                "raw_prob": 0.80,
                "val_accuracy": 0.60,
                "deploy_ok": True,
                "calibrated_prob": 0.80,
            },
        },
        "R_25": {"direction": TradeDirection.PUT, "metrics": {"deploy_ok": False, "gate_reason": "data"}},
        "R_50": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "trade_score": 0.70,
                "raw_prob": 0.70,
                "val_accuracy": 0.60,
                "deploy_ok": True,
                "calibrated_prob": 0.70,
            },
        },
        "R_75": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "trade_score": 0.60,
                "raw_prob": 0.60,
                "val_accuracy": 0.60,
                "deploy_ok": True,
                "calibrated_prob": 0.60,
            },
        },
    }
    with (
        patch(
            "src.application.services.execution_direction_fallback.build_market_execution_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_fallback.build_execution_candidate",
            side_effect=[
                None,
                ("R_10", TradeDirection.CALL, decisions["R_10"]["metrics"]),
                ("R_75", TradeDirection.CALL, decisions["R_75"]["metrics"]),
                ("R_50", TradeDirection.CALL, decisions["R_50"]["metrics"]),
                None,
            ],
        ),
    ):
        picked = _scored_fallback_pick(
            ["R_25", "R_50", "R_10", "R_75"],
            decisions,
            skip_symbols=frozenset({"R_25"}),
            min_signal=0.50,
            min_val=0.50,
        )
        assert picked is not None
        last = _last_resort_fallback_pick(
            ["R_25", "R_50"],
            decisions,
            skip_symbols=frozenset({"R_25"}),
            min_signal=0.50,
            min_val=0.50,
        )
        assert last is not None
        none_build = _last_resort_fallback_pick(
            ["R_75"],
            decisions,
            skip_symbols=frozenset(),
            min_signal=0.50,
            min_val=0.50,
        )
        assert none_build is None
    empty = _last_resort_fallback_pick(
        ["R_25"], decisions, skip_symbols=frozenset({"R_25"}), min_signal=0.50, min_val=0.50
    )
    assert empty is None


def test_binomial_z_tiny_se_returns_zero():
    assert binomial_z_vs_p(10**24, 2 * 10**24) == 0.0
