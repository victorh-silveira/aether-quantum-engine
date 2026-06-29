from src.application.services.execution_quality_gate import passes_execution_quality, quality_gate_params


def test_quality_gate_params_custom_config():
    params = quality_gate_params(
        {"quality_gate": {"min_direction_margin": 0.08, "inverted_min_score": 0.80, "min_adx_normal": 0.20}}
    )
    assert params["min_direction_margin"] == 0.08
    assert params["inverted_min_score"] == 0.80
    assert params["min_adx_normal"] == 0.20


def test_quality_gate_params_defaults():
    params = quality_gate_params({})
    assert params["min_direction_margin"] == 0.05
    assert params["inverted_min_score"] == 0.74
    assert params["min_adx_normal"] == 0.18


def test_passes_execution_quality_rejects_low_edge():
    metrics = {"trade_score": 0.80, "val_accuracy": 0.70, "edge": 0.02, "direction_margin": 0.08}
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_uses_calibrated_edge():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "calibrated_prob": 0.52,
        "calibrated_edge": 0.02,
        "direction_margin": 0.08,
    }
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_uses_dynamic_min_edge():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.05,
        "calibrated_edge": 0.05,
        "dynamic_min_edge": 0.06,
        "direction_margin": 0.08,
    }
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_rejects_low_val():
    metrics = {"trade_score": 0.80, "val_accuracy": 0.50, "edge": 0.10, "direction_margin": 0.08}
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_accepts_complete_signal():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.25},
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        min_adx_normal=0.18,
        recovery_active=False,
    )


def test_passes_execution_quality_rejects_low_signal():
    metrics = {"trade_score": 0.55, "val_accuracy": 0.70, "edge": 0.10, "direction_margin": 0.08}
    assert not passes_execution_quality(metrics, min_signal=0.68, min_val=0.60, min_edge=0.04)


def test_passes_execution_quality_rejects_low_margin():
    metrics = {"trade_score": 0.80, "val_accuracy": 0.70, "edge": 0.10, "direction_margin": 0.02}
    assert not passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        min_direction_margin=0.05,
    )


def test_passes_execution_quality_rejects_inverted_without_strong_signal():
    metrics = {
        "trade_score": 0.70,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "direction_inverted": True,
    }
    assert not passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        inverted_min_score=0.74,
    )


def test_passes_execution_quality_accepts_strong_inverted():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.15,
        "direction_margin": 0.10,
        "direction_inverted": True,
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        inverted_min_score=0.74,
    )


def test_passes_execution_quality_rejects_low_adx_in_normal_mode():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.12},
    }
    assert not passes_execution_quality(
        metrics,
        min_signal=0.68,
        min_val=0.60,
        min_edge=0.04,
        min_adx_normal=0.18,
        recovery_active=False,
    )


def test_passes_execution_quality_skips_adx_check_in_recovery():
    metrics = {
        "trade_score": 0.80,
        "val_accuracy": 0.70,
        "edge": 0.10,
        "direction_margin": 0.08,
        "indicators": {"adx": 0.12},
    }
    assert passes_execution_quality(
        metrics,
        min_signal=0.64,
        min_val=0.60,
        min_edge=0.04,
        min_adx_normal=0.18,
        recovery_active=True,
    )
