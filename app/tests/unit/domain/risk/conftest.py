import json

import pytest

from aether_paths import repo_path


def _full_soft_recovery() -> dict:
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    return dict(full["risk_management"]["soft_recovery"])


def _full_kelly() -> dict:
    path = repo_path("config", "settings.json")
    with path.open(encoding="utf-8") as handle:
        full = json.load(handle)
    return dict(full["risk_management"]["kelly"])


@pytest.fixture
def kelly_config():
    soft = _full_soft_recovery()
    kelly = _full_kelly()
    kelly.update(
        {
            "base_win_rate": 0.50,
            "dynamic_win_rate": True,
            "dynamic_min_samples": 5,
            "fraction": 0.1,
            "max_stake_pct": 0.05,
            "recovery_sizing_conviction": 0.60,
            "recovery_min_conviction": 0.58,
            "recovery_min_val_accuracy": 0.50,
            "stop_win_kelly_enabled": False,
        }
    )
    soft.update(
        {
            "enabled": True,
            "max_safe_stake_cap": 4.20,
            "max_safe_stake_pct": 0.035,
            "amort_cycles_min": 2,
            "amort_cycles_max": 5,
            "coing_redirect_drawdown_threshold": 15.00,
        }
    )
    return {
        "kelly": kelly,
        "soft_recovery": soft,
        "dlambert": {
            "dlambert_enabled": True,
            "recovery_sizing_conviction": 0.60,
            "recovery_min_conviction": 0.58,
            "recovery_min_val_accuracy": 0.50,
        },
        "params": {"payout_estimate": 0.95, "stake_min": 1.0, "entry_cooldown_ticks": 0},
    }
