from src.application.services.execution_regime_gate import apply_regime_boolean_gate, parse_regime_gate_config


def test_parse_regime_gate_config_ssot():
    cfg = parse_regime_gate_config()
    assert cfg["regime_gate_enabled"] is True
    assert float(cfg["regime_adx_max"]) > 0.0
    assert cfg["regime_bb_squeeze_enabled"] is True


def test_regime_gate_hard_skip_preserves_direction(monkeypatch):
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "resolved_direction": "CALL",
        "indicators": {"adx": 0.05, "bb_width": 0.001},
        "micro_indicators": {"bb_width": 0.001},
    }
    monkeypatch.setattr(
        "src.application.services.execution_regime_gate.severe_bb_compression",
        lambda m: True,
    )
    hit = apply_regime_boolean_gate(metrics, force=False)
    assert hit is True
    assert metrics["execution_candidate_ready"] is False
    assert metrics["gate_reason"] == "regime_squeeze"
    assert metrics["gate_verdict"] == "HARD_SKIP"
    assert metrics["exec_direction"] == "CALL"
    assert metrics["resolved_direction"] == "CALL"


def test_regime_gate_noop_when_adx_strong(monkeypatch):
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "PUT",
        "exec_direction": "PUT",
        "indicators": {"adx": 0.40, "bb_width": 0.001},
    }
    monkeypatch.setattr(
        "src.application.services.execution_regime_gate.severe_bb_compression",
        lambda m: True,
    )
    hit = apply_regime_boolean_gate(metrics, force=False)
    assert hit is False
    assert metrics["execution_candidate_ready"] is True


def test_regime_gate_force_and_already_skip():
    metrics = {"execution_candidate_ready": True, "signal_status": "CALL", "indicators": {"adx": 0.01}}
    assert apply_regime_boolean_gate(metrics, force=True) is False
    metrics["signal_status"] = "SKIP:NEG_EDGE"
    assert apply_regime_boolean_gate(metrics, force=False) is False
    metrics = {"execution_candidate_ready": False, "signal_status": "CALL", "indicators": {"adx": 0.01}}
    assert apply_regime_boolean_gate(metrics) is False


def test_regime_gate_missing_adx():
    metrics = {"execution_candidate_ready": True, "signal_status": "CALL", "indicators": {}}
    assert apply_regime_boolean_gate(metrics) is False


def test_regime_gate_disabled():
    metrics = {"execution_candidate_ready": True, "signal_status": "CALL", "indicators": {"adx": 0.01}}
    cfg = parse_regime_gate_config({"regime_gate_enabled": False})
    assert apply_regime_boolean_gate(metrics, cfg=cfg) is False


def test_parse_regime_gate_rejects_bad_adx_max():
    try:
        parse_regime_gate_config({"regime_adx_max": 0.0})
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "regime_adx_max" in str(exc)


def test_regime_gate_invalid_indicator_and_orch_log(monkeypatch):
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "indicators": {"adx": "bad"},
        "micro_indicators": {"adx": object()},
    }
    assert apply_regime_boolean_gate(metrics) is False
    metrics = {
        "execution_candidate_ready": True,
        "signal_status": "CALL",
        "exec_direction": "CALL",
        "indicators": {"adx": 0.05, "bb_width": 0.001},
        "micro_indicators": {"bb_width": 0.001},
    }
    monkeypatch.setattr(
        "src.application.services.execution_regime_gate.severe_bb_compression",
        lambda m: True,
    )
    orch = type("O", (), {})()
    assert apply_regime_boolean_gate(metrics, orch=orch) is True
