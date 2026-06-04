import pytest

from tests.market_symbols import ALL_SYMBOLS, ANCHOR, PAIR


@pytest.fixture
def orch_config():
    return {
        "api_config": {"base_url": "ws://test", "request_timeout_seconds": 1},
        "symbols": list(ALL_SYMBOLS),
        "anchor": ANCHOR,
        "deep_learning": {"enabled": True, "min_conviction_execute": 0.53},
        "data_handler": {"fetch_count": 100, "min_required_points": 2, "buffer_limit": 1000},
        "strategy": {
            "clusters": {"rd": [ANCHOR, PAIR]},
            "correlation": {"anchor": ANCHOR},
        },
        "risk_management": {
            "small_account_threshold": 100.0,
            "small_account_stake": 1.0,
            "small_account_stop_win": 10.0,
            "large_account_stake_pct": 2.0,
            "large_account_stop_win_pct": 15.0,
            "params": {
                "duration": 2,
                "duration_unit": "m",
                "payout_estimate": 0.95,
                "entry_cooldown_ticks": 60,
                "stake_min": 0.5,
                "base_stake_min_pct": 0.01,
                "base_stake_max_pct": 0.02,
            },
            "kelly": {"fraction": 0.5, "base_win_rate": 0.55},
        },
        "orchestrator": {
            "reconcile_interval_seconds": 1,
            "cycle_interval_seconds": 0,
            "execution": {"include_anchor_trades": True, "inter_symbol_delay": 0.25},
        },
        "trading": {"mode": "demo", "session": {"enabled": False}},
    }
