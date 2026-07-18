from src.application.services.deep_learning.dl_params import (
    parse_dl_params,
    resolve_dl_granularity,
    resolve_train_timeframe,
)
from src.application.services.meta_payoff_shadow import (
    record_meta_payoff_shadow_pair,
    reset_meta_payoff_shadow,
)


def test_resolve_train_timeframe_micro_aliases():
    assert resolve_train_timeframe({"train_timeframe": "micro"}) == "micro"
    assert resolve_train_timeframe({"train_timeframe": "settlement"}) == "micro"
    assert resolve_train_timeframe({"train_timeframe": "macro"}) == "macro"
    assert resolve_train_timeframe({}) == "macro"


def test_resolve_dl_granularity_respects_train_timeframe():
    data = {"granularity": 600, "micro_granularity": 120}
    assert resolve_dl_granularity({"train_timeframe": "macro"}, data) == 600
    assert resolve_dl_granularity({"train_timeframe": "micro"}, data) == 120


def test_parse_dl_params_micro_uses_micro_history_bars():
    params = parse_dl_params(
        {"train_timeframe": "micro", "lookback": 72, "label_mode": "spot_forward"},
        {"granularity": 600, "micro_granularity": 120, "micro_history_bars": 5000, "history_bars": 23328},
        {"duration": 120, "duration_unit": "s"},
    )
    assert params["train_timeframe"] == "micro"
    assert params["granularity"] == 120
    assert params["training_history_bars"] == 5000
    assert params["label_horizon_bars"] == 1
    assert params["label_mode"] == "spot_forward"


def test_meta_payoff_shadow_correlation_builds_over_pairs():
    reset_meta_payoff_shadow()
    orch = type("O", (), {})()
    corr = None
    for i in range(16):
        corr = record_meta_payoff_shadow_pair(z_score=float(i), profit=float(i) * 0.5, orch=orch)
    assert corr is not None
    assert corr > 0.9
    assert orch._meta_payoff_shadow_n == 16
    reset_meta_payoff_shadow()


def test_meta_payoff_shadow_returns_none_on_zero_variance():
    reset_meta_payoff_shadow()
    orch = type("O", (), {})()
    corr = None
    for _ in range(16):
        corr = record_meta_payoff_shadow_pair(z_score=1.0, profit=0.0, orch=orch)
    assert corr is None
    reset_meta_payoff_shadow()
