"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
import logging
from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.application.services.execution_direction_checks import initial_direction_checks, meta_zscore_soft_ok
from src.application.services.execution_market_rank import mandatory_pool_eligible, market_decision_score
from src.application.services.log_dedupe import LogDeduper, clear_log_channel
from src.application.services.market_audit_log_helpers import cluster_symbol_token
from src.application.services.meta_payoff_shadow import (
    meta_hard_veto_allowed,
    meta_inverted_shadow_active,
    record_meta_payoff_shadow_pair,
    reset_meta_payoff_shadow,
)
from src.application.services.orchestrator.execution_blockers import log_execution_blockers
from src.application.services.orchestrator.orchestrator_data_signature import (
    get_data_state_signature,
    resolve_signature_boundary_seconds,
)
from src.application.services.orchestrator.trading_cycle_entry_guards import (
    _cycle_cadence_elapsed,
    cycle_cadence_seconds,
)
from src.domain.models.trade import TradeDirection


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_market_rank_and_direction_checks_force():
    entry = {"metrics": {"deploy_ok": True, "raw_prob": 0.7}}
    with (
        patch(
            "src.application.services.execution_market_rank.infer_dl_direction",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_market_rank.force_trade_every_cycle",
            return_value=True,
        ),
        patch(
            "src.application.services.execution_market_rank.synthesize_force_direction",
            return_value=TradeDirection.CALL,
        ),
    ):
        assert mandatory_pool_eligible(entry, exec_cfg={"force_trade_every_cycle": True}) is True
    soft_metrics = {"meta_payoff_edge_zscore": -0.1}
    assert meta_zscore_soft_ok(soft_metrics, risk_manager=None) is True
    exec_cfg = {"force_trade_every_cycle": True}
    with (
        patch(
            "src.application.services.execution_direction_checks.infer_dl_direction",
            return_value=None,
        ),
        patch(
            "src.application.services.execution_direction_checks.synthesize_force_direction",
            return_value=TradeDirection.PUT,
        ),
    ):
        out = initial_direction_checks(
            {"metrics": {"deploy_ok": True, "raw_prob": 0.3}},
            exec_cfg,
        )
    assert out is not None
    assert out[0] == TradeDirection.PUT
    metrics_brier = {
        "live_n": 50,
        "live_brier": 0.5,
        "val_brier": 0.2,
        "trade_score": 0.6,
        "edge": 0.1,
        "execute": True,
        "deploy_ok": True,
        "direction_margin": 0.2,
        "indicators": {},
    }
    score = market_decision_score(metrics_brier, recovery_active=False)
    assert isinstance(score, float)


def test_execution_blockers_deploy_ready_force_skip():
    executor = MagicMock()
    executor.orch = SimpleNamespace(
        _active_cycle_id=3,
        config={"orchestrator": {"execution": {"force_trade_every_cycle": True}}},
    )
    executor._trade_symbols = MagicMock(return_value=["OTC_SPC"])
    log_execution_blockers(executor, {"OTC_SPC": {"metrics": {"deploy_ok": False}}})
    executor.orch.config = {"orchestrator": {"execution": {}}}
    log_execution_blockers(
        executor,
        {
            "OTC_SPC": {"metrics": {"execution_candidate_ready": True}},
            "R_50": {"metrics": {"signal_status": "SIGNAL_SUSPENDED"}},
        },
    )


def test_log_deduper_and_market_audit_helpers():
    owner = SimpleNamespace()
    logger = logging.getLogger("test.dedupe")
    deduper = LogDeduper(owner)
    deduper.log_quality_guard_cycle_minute(logger, cycle_id=1, minute_bucket="m", message="q")
    deduper.log_quality_starvation_escape(logger, skipped_cycles=2, min_margin=0.04)
    deduper.log_cooldown_cooling_down(logger, "cool", 1.0, 0)
    deduper.log_cooldown_skip(logger, "skip")
    owner._log_dedupe = {"ch": "x"}
    assert clear_log_channel(owner, "ch") == "x"
    assert cluster_symbol_token("", None) == "N/A"
    assert cluster_symbol_token("r_10", None) == "R_10"
    assert cluster_symbol_token("otc_spc", None) == "OTC_SPC"


def test_meta_payoff_shadow_inverted_and_hard():
    reset_meta_payoff_shadow()
    orch = SimpleNamespace(_meta_payoff_shadow_corr=-0.5, _meta_payoff_shadow_n=20)
    for i in range(20):
        record_meta_payoff_shadow_pair(z_score=float(i), profit=-float(i), orch=orch)
    with patch(
        "src.application.services.meta_payoff_shadow._shadow",
        return_value={
            "min_pairs": 8,
            "ready_n": 8,
            "window": 64,
            "hard_corr_floor": 0.5,
            "soft_only_corr_ceiling": 0.2,
        },
    ):
        assert meta_inverted_shadow_active(orch) is True
        reset_meta_payoff_shadow()
        for i in range(12):
            record_meta_payoff_shadow_pair(z_score=float(i), profit=float(i), orch=None)
        assert meta_hard_veto_allowed(None) is True


def test_orchestrator_data_signature_bad_config():
    orch = SimpleNamespace(config="bad", symbols=[], stream=None)
    assert resolve_signature_boundary_seconds(orch) == 60
    orch2 = SimpleNamespace(
        config={"orchestrator": {"signature_boundary_seconds": "bad", "cycle_interval_seconds": "x"}},
        symbols=[],
        stream=SimpleNamespace(micro_candles={}, macro_candles={}),
    )
    assert resolve_signature_boundary_seconds(orch2) == 60
    assert get_data_state_signature(orch2) == ""


def test_trading_cycle_cadence_exec_empty():
    orch = SimpleNamespace(
        config={"orchestrator": {"cycle_interval_seconds": 120, "exec_empty_retry_seconds": 30}},
        _last_cycle_was_exec_empty=True,
    )
    assert cycle_cadence_seconds(orch) == 30
    orch2 = SimpleNamespace(config={"orchestrator": {"cycle_interval_seconds": 0}}, _last_cluster_cycle_end=0.0)
    assert _cycle_cadence_elapsed(orch2) is False
