from src.application.services.deep_learning.dl_params import (
    parse_dl_params,
    resolve_dl_granularity,
    resolve_train_timeframe,
)


def test_resolve_train_timeframe_micro_aliases():
    assert resolve_train_timeframe({"train_timeframe": "micro"}) == "micro"
    assert resolve_train_timeframe({"train_timeframe": "settlement"}) == "micro"
    assert resolve_train_timeframe({"train_timeframe": "macro"}) == "macro"
    assert resolve_train_timeframe({}) == "micro"


def test_resolve_dl_granularity_respects_train_timeframe():
    data = {"granularity": 300, "micro_granularity": 60}
    assert resolve_dl_granularity({"train_timeframe": "macro"}, data) == 300
    assert resolve_dl_granularity({"train_timeframe": "micro"}, data) == 60


def test_parse_dl_params_micro_uses_micro_history_bars():
    params = parse_dl_params(
        {"train_timeframe": "micro", "lookback": 72, "label_mode": "spot_forward"},
        {"granularity": 300, "micro_granularity": 60, "micro_history_bars": 5000, "history_bars": 23328},
        {"duration": 30, "duration_unit": "s"},
    )
    assert params["train_timeframe"] == "micro"
    assert params["granularity"] == 60
    assert params["training_history_bars"] == 5000
    assert params["label_horizon_bars"] == 1
    assert params["label_mode"] == "spot_forward"
    assert params["contract_duration"] == 30
