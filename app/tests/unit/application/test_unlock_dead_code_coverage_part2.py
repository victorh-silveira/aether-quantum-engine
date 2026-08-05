from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.orchestrator.execution_collect_helpers import revive_ready_cluster_candidates
from src.domain.models.trade import TradeDirection


def test_shrink_toward_fifty_below_floor():
    from src.application.services.deep_learning.dl_calibration import shrink_toward_fifty

    result = shrink_toward_fifty(0.55, 0.005)
    assert 0.50 < result < 0.55


def test_rolling_zscore_fast_empty():
    from src.application.services.deep_learning.dl_feature_oscillators import rolling_zscore_fast

    result = rolling_zscore_fast(np.array([], dtype=np.float64), window=14, clip=3.0)
    assert len(result) == 0


def test_schedule_model_upload_exception_caught():
    from src.application.services.deep_learning.dl_model_artifacts import schedule_model_upload

    class _FakeLoop:
        def is_running(self):
            return True

        @staticmethod
        def call_soon_threadsafe(_cb):
            raise RuntimeError("event loop closed")

    orch = SimpleNamespace(infra=SimpleNamespace(enabled=True), loop=_FakeLoop())
    with patch("src.application.services.deep_learning.dl_model_artifacts.upload_model_checkpoint"):
        schedule_model_upload(orch, "R_10", Path("checkpoint.pt"), arch="tcn")


def test_schedule_model_upload_success_return():
    from src.application.services.deep_learning.dl_model_artifacts import schedule_model_upload

    class _OkLoop:
        def is_running(self):
            return True

        @staticmethod
        def call_soon_threadsafe(_cb):
            return

    orch = SimpleNamespace(infra=SimpleNamespace(enabled=True), loop=_OkLoop())
    with patch("src.application.services.deep_learning.dl_model_artifacts.upload_model_checkpoint"):
        schedule_model_upload(orch, "R_10", Path("checkpoint.pt"), arch="tcn")


def test_revive_ready_keeps_side_eq_blocked_sizing_only():
    exec_mgr = MagicMock()
    exec_mgr._trade_symbols.return_value = ["R_10"]
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execution_candidate_ready": True,
                "side_eq_blocked": True,
                "exec_direction": "CALL",
            },
        }
    }
    result = revive_ready_cluster_candidates(exec_mgr, decisions)
    assert len(result) == 1
    assert result[0][0] == "R_10"


def test_settlement_timed_out_true():
    from src.application.services.orchestrator.execution_settlement import _settlement_timed_out

    exec_mgr = MagicMock()
    exec_mgr.logger = MagicMock()
    exec_mgr.orch.risk_manager.active_contract_ids = []
    with (
        patch("src.application.services.orchestrator.execution_settlement.time.time", return_value=999),
        patch("src.application.services.orchestrator.execution_settlement.settlement_utils"),
    ):
        result = _settlement_timed_out(exec_mgr, start_time=0, timeout=10)
    assert result is True
    exec_mgr.logger.error.assert_called_once_with("EXEC: Timeout fatal aguardando liquidacao.")


@pytest.mark.asyncio
async def test_clean_stale_redis_exception():
    from src.application.services.orchestrator.post_settlement_cycle import _clean_stale_settlement_and_redis_counters

    orch = MagicMock()
    orch.logger = MagicMock()
    with patch(
        "src.application.services.orchestrator.post_settlement_cycle.get_redis_client",
        side_effect=Exception("redis connection refused"),
    ):
        await _clean_stale_settlement_and_redis_counters(orch)
    orch.logger.error.assert_called_once()


def test_check_session_limits_no_state_mgr_with_target_zero():
    from src.application.services.orchestrator.settlement_outcome import check_session_limits_before_post_settlement

    orch = SimpleNamespace(
        risk_manager=SimpleNamespace(total_session_profit=0.0, initial_bankroll=0.0),
        config={"risk_management": {}},
        state_mgr=None,
    )
    result = check_session_limits_before_post_settlement(orch)
    assert result is False


def test_update_state_manager_no_state_mgr():
    from src.application.services.orchestrator.settlement_outcome import update_state_manager_and_check_stop_win

    orch = SimpleNamespace(state_mgr=None)
    result = update_state_manager_and_check_stop_win(orch, target=50.0, pnl=100.0)
    assert result is True


def test_stop_win_blocks_cycle_target_zero():
    from src.application.services.orchestrator.trading_cycle_entry_guards import _stop_win_blocks_cycle

    orch = SimpleNamespace(
        shutdown_reason=None,
        risk_manager=SimpleNamespace(initial_bankroll=0.0, total_session_profit=0.0),
        config={},
        state_mgr=SimpleNamespace(state=SimpleNamespace(daily_stop_win_target=0.0)),
    )
    with patch(
        "src.application.services.orchestrator.trading_cycle_entry_guards.resolve_stop_win_target", return_value=0.0
    ):
        result = _stop_win_blocks_cycle(orch)
    assert result is False


def test_apply_small_account_hard_floor_below_threshold():
    from src.domain.risk.soft_recovery_policy import apply_small_account_hard_floor

    result = apply_small_account_hard_floor(cap=1000.0, bankroll=0.005)
    assert result == 5e-05


def test_recovery_infeasible_log():
    from src.domain.risk.risk_stake_calc import calculate_stake_for_manager

    rm = MagicMock()
    rm.pending_loss = {"R_10": 100.0}
    rm.consecutive_losses_linear = 0
    rm.dlambert_unit = 0.0
    rm.kelly_config = {"mandatory_weak_conviction_cap": 0.55, "recovery_min_conviction": 0.50}
    rm.risk_params = {"stake_min": 1.0}
    rm.logger = MagicMock()
    rm.dlambert_config = {}

    kwargs = {"dl_metrics": {"recovery_infeasible": True, "execute": True}}
    with (
        patch("src.domain.risk.risk_stake_calc.check_stake_preconditions_veto", return_value=False),
        patch("src.domain.risk.risk_stake_calc.clear_dust_pending_loss"),
        patch("src.domain.risk.risk_stake_calc.resolve_stake_regime", return_value="EXPLORE"),
        patch("src.domain.risk.risk_stake_calc.resolve_stake_conviction", return_value=0.55),
        patch("src.domain.risk.risk_stake_calc.calculate_kelly_fraction", return_value=(0.5, 0.95, 1.0)),
        patch("src.domain.risk.risk_stake_calc.d_squeeze_sovereignty_active", return_value=False),
        patch("src.domain.risk.risk_stake_calc._resolve_recovery_flags", return_value=(False, False, False, 0)),
        patch("src.domain.risk.risk_stake_calc.resolve_f_star_and_kelly_base", return_value=(0.5, 100.0)),
        patch(
            "src.domain.risk.risk_stake_calc._apply_stop_win_kelly_boost", side_effect=lambda rm, **kw: kw["kelly_base"]
        ),
        patch("src.domain.risk.risk_stake_calc.resolve_dlambert_stake", return_value=(5.0, "EXPLORE")),
        patch("src.domain.risk.risk_stake_calc._mandatory_trade_flag", return_value=False),
        patch("src.domain.risk.risk_stake_calc.apply_turbo_edge_stake", side_effect=lambda fs, dm: fs),
        patch("src.domain.risk.risk_stake_calc.cap_final_stake", return_value=(5.0, 1000.0)),
        patch("src.domain.risk.risk_stake_calc._apply_mandatory_weak_explore_cap", side_effect=lambda fs, br, **kw: fs),
        patch("src.domain.risk.risk_stake_calc.finalize_stake_with_min", side_effect=lambda fs, sm, br, cv, **kw: fs),
        patch("src.domain.risk.risk_stake_calc.enforce_d_squeeze_stake_floor", side_effect=lambda fs, sm, dm, **kw: fs),
        patch("src.domain.risk.risk_stake_calc.dlambert_log_suffix"),
        patch("src.domain.risk.risk_stake_calc.effective_soft_recovery_base"),
    ):
        calculate_stake_for_manager(
            rm, bankroll=1000.0, symbol="R_10", conviction=0.65, silent=False, apply_stop_win=False, kwargs=kwargs
        )

    rm.logger.info.assert_called()
    first_call = rm.logger.info.call_args_list[0]
    assert "RECOVERY_INFEASIBLE" in first_call[0][0]


def test_horizon_gap_bars():
    from src.application.services.deep_learning.dl_calibration_tolerance import _horizon_gap_bars

    result = _horizon_gap_bars()
    assert isinstance(result, int)
    assert result >= 0


def test_in_neutral_zone_none_branches():
    from src.application.services.deep_learning.dl_calibration_tolerance import _in_neutral_zone

    assert _in_neutral_zone(0.5, None, None) is False
    assert _in_neutral_zone(0.5, 0.4, None) is False
    assert _in_neutral_zone(0.5, None, 0.6) is False
