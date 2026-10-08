from src.application.services.orchestrator.engine_mode import (
    ENGINE_MODE_EXECUTE,
    ENGINE_MODE_TRAIN,
    apply_engine_mode,
    resolve_engine_mode,
    training_enabled,
)


def test_resolve_engine_mode_defaults_to_execute(orch_config):
    assert resolve_engine_mode(orch_config) == ENGINE_MODE_EXECUTE


def test_resolve_engine_mode_without_orchestrator_dict():
    assert resolve_engine_mode({}) == ENGINE_MODE_EXECUTE


def test_resolve_engine_mode_train(orch_config):
    orch_config.setdefault("orchestrator", {})["engine_mode"] = "train"
    assert resolve_engine_mode(orch_config) == ENGINE_MODE_TRAIN


def test_resolve_engine_mode_training_alias(orch_config):
    orch_config.setdefault("orchestrator", {})["engine_mode"] = "training"
    assert resolve_engine_mode(orch_config) == ENGINE_MODE_TRAIN


def test_apply_engine_mode(orch_config):
    apply_engine_mode(orch_config, ENGINE_MODE_TRAIN)
    assert orch_config["orchestrator"]["engine_mode"] == ENGINE_MODE_TRAIN


def test_apply_engine_mode_aligns_label_horizon_to_micro_contract():
    config = {
        "orchestrator": {},
        "data_handler": {"granularity": 300, "micro_granularity": 60},
        "deep_learning": {"train_timeframe": "micro", "label_horizon_bars": 99},
        "risk_management": {"params": {"duration": 30, "duration_unit": "s"}},
    }
    apply_engine_mode(config, ENGINE_MODE_TRAIN)
    assert config["deep_learning"]["label_horizon_bars"] == 1


def test_apply_engine_mode_aligns_label_horizon_when_contract_spans_bars():
    config = {
        "orchestrator": {},
        "data_handler": {"granularity": 60, "micro_granularity": 60},
        "deep_learning": {"train_timeframe": "macro", "label_horizon_bars": 99},
        "risk_management": {"params": {"duration": 5, "duration_unit": "m"}},
    }
    apply_engine_mode(config, ENGINE_MODE_TRAIN)
    assert config["deep_learning"]["label_horizon_bars"] == 5


def test_apply_engine_mode_aligns_r10_m3_nine_minute_horizon():
    config = {
        "orchestrator": {},
        "data_handler": {"granularity": 7200, "micro_granularity": 180},
        "deep_learning": {"train_timeframe": "micro", "label_horizon_bars": 1},
        "risk_management": {"params": {"duration": 9, "duration_unit": "m"}},
    }
    apply_engine_mode(config, ENGINE_MODE_TRAIN)
    assert config["deep_learning"]["label_horizon_bars"] == 3


def test_apply_engine_mode_aligns_r10_m3_fifteen_minute_horizon():
    config = {
        "orchestrator": {},
        "data_handler": {"granularity": 7200, "micro_granularity": 180},
        "deep_learning": {"train_timeframe": "micro", "label_horizon_bars": 1},
        "risk_management": {"params": {"duration": 15, "duration_unit": "m"}},
    }
    apply_engine_mode(config, ENGINE_MODE_TRAIN)
    assert config["deep_learning"]["label_horizon_bars"] == 5


def test_apply_engine_mode_replaces_invalid_orchestrator():
    config: dict = {"orchestrator": "invalid"}
    apply_engine_mode(config, ENGINE_MODE_TRAIN)
    assert config["orchestrator"]["engine_mode"] == ENGINE_MODE_TRAIN


def test_training_enabled(orch_ready):
    apply_engine_mode(orch_ready.config, ENGINE_MODE_TRAIN)
    assert training_enabled(orch_ready) is True
    apply_engine_mode(orch_ready.config, ENGINE_MODE_EXECUTE)
    assert training_enabled(orch_ready) is False
