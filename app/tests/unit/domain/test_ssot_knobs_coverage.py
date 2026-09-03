import json
from io import StringIO

import pytest

import src.domain.risk.kelly_runtime_config as kelly_rt_mod
import src.domain.risk.recovery_state_config as rec_state_mod
import src.domain.risk.soft_recovery_config as soft_cfg_mod
import src.domain.risk.soft_recovery_policy as soft_pol_mod
from src.domain.risk.consensus_stake_helpers import _runtime as consensus_runtime
from src.domain.risk.kelly_base_fraction import resolve_effective_kelly_fraction
from src.domain.risk.kelly_runtime_config import (
    kelly_runtime_from_config,
    load_kelly_runtime_from_settings,
    reset_kelly_runtime_config_cache,
)
from src.domain.risk.recovery_conviction import recovery_min_conviction
from src.domain.risk.recovery_state_config import (
    load_recovery_state_from_settings,
    reset_recovery_state_config_cache,
    resolve_recovery_state_config,
)
from src.domain.risk.risk_recovery_state import cointegration_pair_score
from src.domain.risk.risk_stake_calc import _apply_mandatory_weak_explore_cap
from src.domain.risk.soft_recovery_config import (
    load_soft_recovery_from_settings,
    reset_soft_recovery_config_cache,
    resolve_soft_recovery_config,
)
from src.domain.risk.soft_recovery_policy import configured_max_safe_stake_pct
from src.domain.risk.stake_target_proximity import resolve_target_proximity_damping


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def _incomplete_kelly(**extra):
    return {"neutral_bankroll_pct": 0.01, "recovery_conviction_ladder": {"losses_1": 0.5}, **extra}


def test_soft_recovery_reset_and_flat_override():
    reset_soft_recovery_config_cache()
    assert soft_cfg_mod._CACHE["soft_recovery"] is None
    base = load_soft_recovery_from_settings()
    flat = resolve_soft_recovery_config({"enabled": True, "max_safe_stake_cap": base["max_safe_stake_cap"]})
    assert flat["enabled"] is True
    assert flat["max_safe_stake_cap"] == pytest.approx(base["max_safe_stake_cap"])


def test_configured_max_safe_stake_pct_bad_float(monkeypatch):
    monkeypatch.setattr(soft_pol_mod, "soft_cfg", lambda *_a, **_k: {"max_safe_stake_pct": object()})
    assert configured_max_safe_stake_pct(None) == pytest.approx(
        load_soft_recovery_from_settings()["max_safe_stake_pct"]
    )


def test_kelly_runtime_reset_and_from_config_paths(monkeypatch):
    reset_kelly_runtime_config_cache()
    assert kelly_rt_mod._CACHE["kelly_runtime"] is None
    base = load_kelly_runtime_from_settings()
    assert kelly_runtime_from_config({"kelly": base})["fraction"] == base["fraction"]
    assert kelly_runtime_from_config({"risk_management": {"kelly": base}})["fraction"] == base["fraction"]
    assert kelly_runtime_from_config({})["fraction"] == base["fraction"]
    assert kelly_runtime_from_config(None)["fraction"] == base["fraction"]


def test_kelly_base_fraction_and_consensus_runtime_fallback():
    out = resolve_effective_kelly_fraction(_incomplete_kelly(fraction_base_retention=0.4, fraction=0.01))
    assert out > 0.0
    rt = consensus_runtime(_incomplete_kelly())
    assert "payout_fallback" in rt


def test_recovery_conviction_incomplete_ladder_fallback():
    floor = recovery_min_conviction(
        _incomplete_kelly(recovery_sizing_conviction=0.55),
        {},
        pending_loss={},
        consecutive_losses_linear=0,
    )
    assert floor > 0.0


def test_recovery_state_resolve_reset_and_missing(monkeypatch):
    reset_recovery_state_config_cache()
    base = load_recovery_state_from_settings()
    assert resolve_recovery_state_config({"recovery_state": dict(base)})["raw_prob_default"] == base["raw_prob_default"]
    assert resolve_recovery_state_config(None)["micro_unit_floor"] == base["micro_unit_floor"]
    reset_recovery_state_config_cache()
    assert rec_state_mod._CACHE["recovery_state"] is None
    monkeypatch.setattr(
        "settings_io.load_settings_json",
        lambda: {"risk_management": {}},
    )
    with pytest.raises(ValueError, match="recovery_state"):
        load_recovery_state_from_settings()
    reset_recovery_state_config_cache()


def test_stake_target_proximity_with_partial_kelly_config():
    runtime = load_kelly_runtime_from_settings()
    both = resolve_target_proximity_damping(
        100.0,
        50.0,
        kelly_config={"target_damping_floor": 0.4, "target_damping_span": 0.6},
    )
    assert both == pytest.approx(0.7)
    floor_only = resolve_target_proximity_damping(100.0, 0.0, kelly_config={"target_damping_floor": 0.5})
    assert floor_only == pytest.approx(0.5 + float(runtime["target_damping_span"]))
    span_only = resolve_target_proximity_damping(100.0, 0.0, kelly_config={"target_damping_span": 0.5})
    assert span_only == pytest.approx(float(runtime["target_damping_floor"]) + 0.5)


def test_cointegration_pair_score_raw_and_default_prob():
    assert cointegration_pair_score({"raw_prob": 0.7, "edge_zscore": 1.2}) > float("-inf")
    assert cointegration_pair_score({"edge_zscore": 1.2}) > float("-inf")


def test_mandatory_weak_explore_cap_zero_pct():
    assert (
        _apply_mandatory_weak_explore_cap(
            10.0,
            1000.0,
            stake_regime="EXPLORE",
            mandatory_flag=True,
            dl_execute=False,
            kelly_config={},
        )
        == 10.0
    )
