from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.deep_learning.dl_calibration import CalibratorState, apply_calibrator_stable
from src.application.services.execution_direction import build_execution_candidate, mandatory_execution_eligible
from src.application.services.execution_direction_fallback import (
    _last_resort_fallback_pick,
    _scored_fallback_pick,
    build_mandatory_fallback_candidate,
)
from src.application.services.execution_entropy_fallback import pick_entropy_fallback_candidate
from src.application.services.execution_symbols import select_mandatory_execution_candidate
from src.domain.analytics.side_equilibrium import binomial_z_vs_p
from src.domain.models.trade import TradeDirection


def test_calibrator_binomial_mandatory_peer_and_pool():
    cal = CalibratorState(method="isotonic", isotonic_x=(0.2, 0.8), isotonic_y=(0.1, 0.9))
    assert apply_calibrator_stable(0.05, cal) == pytest.approx(0.05)
    assert apply_calibrator_stable(0.95, cal) == pytest.approx(0.95)
    with patch(
        "src.application.services.deep_learning.dl_calibration.apply_calibrator",
        return_value=0.2,
    ):
        assert apply_calibrator_stable(0.7, cal) == pytest.approx(0.7)
    assert binomial_z_vs_p(10**24, 10**24) == 0.0
    assert mandatory_execution_eligible({"metrics": {"deploy_ok": True}}) is False
    assert mandatory_execution_eligible({"metrics": {"deploy_ok": True, "calibrated_prob": 0.7}}) is False
    with (
        patch("src.application.services.execution_direction.hedge_peer", return_value="R_25"),
        patch(
            "src.application.services.execution_direction.resolve_execution_direction",
            return_value=None,
        ) as resolve,
    ):
        assert (
            build_execution_candidate(
                "R_10",
                {"metrics": {}},
                decisions={"R_25": {"metrics": {"x": 1}}},
            )
            is None
        )
        assert resolve.call_args.kwargs.get("peer_entry") == {"metrics": {"x": 1}}
    assert (
        select_mandatory_execution_candidate(
            SimpleNamespace(config={}), [], last_loss_symbol=None, recovery_active=False
        )
        is None
    )


def test_entropy_and_fallback_paths():
    orch = SimpleNamespace(_active_cycle_id=4, config={})
    good = {
        "calibrated_prob": 0.7,
        "raw_prob": 0.7,
        "deploy_ok": True,
        "execute": True,
        "predicted_payoff_edge": 0.1,
        "meta_classifier_applied": True,
        "val_accuracy": 0.7,
        "trade_score": 0.8,
        "conviction": 0.8,
    }
    mid = {**good, "trade_score": 0.55, "conviction": 0.55, "calibrated_prob": 0.55, "raw_prob": 0.55}
    weak = {**good, "trade_score": 0.4, "conviction": 0.4, "calibrated_prob": 0.4, "raw_prob": 0.4}
    decisions = {
        "SKIP": {"direction": TradeDirection.CALL, "metrics": good},
        "R_10": {"direction": TradeDirection.CALL, "metrics": good},
        "R_50": {"direction": TradeDirection.CALL, "metrics": mid},
        "R_25": {"direction": TradeDirection.PUT, "metrics": weak},
        "DEAD": {"direction": TradeDirection.CALL, "metrics": good},
    }
    candidate = ("R_10", TradeDirection.CALL, good)
    mid_candidate = ("R_50", TradeDirection.CALL, mid)
    with (
        patch(
            "src.application.services.execution_entropy_fallback.is_technically_blocked",
            return_value=False,
        ),
        patch(
            "src.application.services.execution_entropy_fallback.build_execution_candidate",
            return_value=candidate,
        ),
    ):
        assert pick_entropy_fallback_candidate(["R_10"], decisions, orch=orch, cycle_id=0) is not None

    def _build_by_symbol(symbol, entry, **kwargs):
        if symbol == "DEAD":
            return None
        if symbol == "R_50":
            return mid_candidate
        return candidate

    with (
        patch(
            "src.application.services.execution_direction_fallback.build_market_execution_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_fallback.build_execution_candidate",
            side_effect=_build_by_symbol,
        ),
    ):
        scored = _scored_fallback_pick(
            ["SKIP", "R_25", "R_10", "R_50"],
            decisions,
            skip_symbols=frozenset({"SKIP"}),
            min_signal=0.5,
            min_val=0.5,
            orch=orch,
        )
        assert scored is not None
        assert scored[0] == "R_10"
        last = _last_resort_fallback_pick(
            ["SKIP", "DEAD", "R_10"],
            decisions,
            skip_symbols=frozenset({"SKIP"}),
            min_signal=0.5,
            min_val=0.5,
            orch=orch,
        )
        assert last is not None
    with (
        patch(
            "src.application.services.execution_direction_fallback.pick_best_mandatory_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_fallback._scored_fallback_pick",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_fallback._last_resort_fallback_pick",
            return_value=candidate,
        ),
    ):
        assert (
            build_mandatory_fallback_candidate(
                ["R_10"],
                decisions,
                recovery_active=False,
                last_loss_symbol=None,
                orch=orch,
            )
            == candidate
        )
