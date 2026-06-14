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


def test_apply_engine_mode_replaces_invalid_orchestrator():
    config: dict = {"orchestrator": "invalid"}
    apply_engine_mode(config, ENGINE_MODE_TRAIN)
    assert config["orchestrator"]["engine_mode"] == ENGINE_MODE_TRAIN


def test_training_enabled(orch_ready):
    apply_engine_mode(orch_ready.config, ENGINE_MODE_TRAIN)
    assert training_enabled(orch_ready) is True
    apply_engine_mode(orch_ready.config, ENGINE_MODE_EXECUTE)
    assert training_enabled(orch_ready) is False
