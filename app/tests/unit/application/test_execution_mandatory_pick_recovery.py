from src.application.services.execution_mandatory_pick import _recovery_hedge_pick, pick_best_mandatory_candidate
from src.application.services.execution_market_rank import resolve_market_direction
from src.domain.models.trade import TradeDirection


def test_recovery_hedge_skips_blocked_peer():
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"trade_score": 0.62, "raw_prob": 0.58, "deploy_ok": True},
        },
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.70, "raw_prob": 0.42, "deploy_ok": True},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_50"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_100",
        last_loss_direction="PUT",
        skip_symbols=frozenset({"R_10"}),
    )
    assert picked is not None
    assert picked[0] != "R_10"


def test_recovery_hedge_resolves_direction_without_raw_prob():
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {
                "deploy_ok": True,
                "binary_ctx": {"body": 0.001, "close_loc": 0.55, "sma_z": 0.0},
            },
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_100",
        last_loss_direction="PUT",
    )
    assert picked is not None
    assert picked[0] == "R_10"
    assert picked[1] == TradeDirection.CALL


def test_recovery_hedge_pick_returns_none_without_resolvable_direction():
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {"deploy_ok": True},
        },
    }
    picked = _recovery_hedge_pick(
        decisions,
        last_loss_symbol="R_100",
        last_loss_direction="PUT",
        skip_symbols=frozenset(),
    )
    assert picked is None


def test_recovery_hedge_returns_none_when_peer_has_no_direction():
    decisions = {
        "R_10": {
            "direction": None,
            "metrics": {"deploy_ok": True},
        },
        "R_50": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.70, "raw_prob": 0.42, "deploy_ok": True},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_50"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_100",
        last_loss_direction="PUT",
    )
    assert picked is not None
    assert picked[0] == "R_50"


def test_pick_best_recovery_uses_range_hedge_peer():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.62, "raw_prob": 0.44, "deploy_ok": True, "execute": True},
        },
        "R_100": {
            "direction": TradeDirection.PUT,
            "metrics": {"trade_score": 0.70, "raw_prob": 0.42, "deploy_ok": True, "execute": True},
        },
    }
    picked = pick_best_mandatory_candidate(
        ["R_10", "R_100"],
        decisions,
        recovery_active=True,
        last_loss_symbol="R_100",
        last_loss_direction="PUT",
    )
    assert picked is not None
    assert picked[0] == "R_10"
    assert picked[1] == TradeDirection.CALL


def test_resolve_market_direction_weak_without_ctx_keeps_dl():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "trade_score": 0.0,
            "raw_prob": 0.49,
            "gate_reason": "raw_conviction",
        },
    }
    assert resolve_market_direction(entry) == TradeDirection.PUT
