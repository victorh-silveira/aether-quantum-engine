import pytest


@pytest.fixture
def kelly_config():
    return {
        "kelly": {
            "base_win_rate": 0.50,
            "dynamic_win_rate": True,
            "dynamic_min_samples": 5,
            "fraction": 0.1,
            "max_stake_pct": 0.05,
            "martingale_sizing_conviction": 0.60,
            "stop_win_kelly_enabled": False,
        },
        "params": {"payout_estimate": 0.95, "stake_min": 1.0, "entry_cooldown_ticks": 0},
    }
