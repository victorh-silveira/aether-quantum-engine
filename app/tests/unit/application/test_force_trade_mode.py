from types import SimpleNamespace

from src.application.services.force_trade_mode import (
    force_trade_every_cycle,
    force_trade_from_config,
    force_trade_from_orch,
    resolve_force_min_stake,
    synthesize_force_direction,
    synthesize_force_trade_candidate,
)
from src.domain.models.trade import TradeDirection


def test_force_trade_flags():
    assert force_trade_every_cycle(None) is False
    assert force_trade_every_cycle({"force_trade_every_cycle": True}) is True
    assert force_trade_from_config(None) is False
    assert force_trade_from_config({"orchestrator": {"execution": {"force_trade_every_cycle": True}}}) is True
    assert force_trade_from_orch(None) is False
    assert (
        force_trade_from_orch(
            SimpleNamespace(config={"orchestrator": {"execution": {"force_trade_every_cycle": False}}})
        )
        is False
    )


def test_resolve_force_min_stake_paths():
    assert resolve_force_min_stake(None) > 0.0
    assert resolve_force_min_stake({"risk_management": {}}) > 0.0
    assert resolve_force_min_stake({"risk_management": {"params": {"stake_min": 1.25}}}) >= 1.25
    assert resolve_force_min_stake({"risk_management": {"params": {"stake_min": "bad"}}}) > 0.0


def test_synthesize_force_direction_and_candidate():
    assert synthesize_force_direction({"metrics": {"deploy_ok": False}}) is None
    assert synthesize_force_direction({"metrics": {"gate_reason": "training"}}) is None
    direction = synthesize_force_direction({"metrics": {"calibrated_prob": 0.7, "deploy_ok": True}})
    assert direction == TradeDirection.CALL
    candidate = synthesize_force_trade_candidate(
        ["OTC_SPC"],
        {"OTC_SPC": {"metrics": {"calibrated_prob": 0.3, "deploy_ok": True}}},
    )
    assert candidate is not None
