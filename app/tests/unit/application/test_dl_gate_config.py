from src.application.services.deep_learning.dl_gate_config import (
    deploy_params_for_eval,
    parse_deploy_gate_config,
    resolve_deploy_ok,
)


def test_resolve_deploy_ok_soft_fallback():
    cfg = parse_deploy_gate_config(
        {"deploy_gate": {"enabled": True, "soft_min_val_accuracy": 0.50, "soft_max_brier": 0.32}}
    )
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.52, val_brier=0.25, gate_cfg=cfg) is True
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.48, val_brier=0.25, gate_cfg=cfg) is False


def test_resolve_deploy_ok_force_ok():
    cfg = parse_deploy_gate_config({"deploy_gate": {"enabled": True, "force_ok": True}})
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.40, val_brier=0.30, gate_cfg=cfg) is True


def test_deploy_params_for_eval_relaxes_thresholds():
    params = {
        "confidence_call_threshold": 0.75,
        "confidence_put_threshold": 0.25,
        "min_val_accuracy": 0.52,
    }
    cfg = {"eval_relaxed_gating": True}
    out = deploy_params_for_eval(params, cfg)
    assert out["min_val_accuracy"] == 0.0
    assert out["confidence_call_threshold"] <= 0.65


def test_deploy_params_passthrough_when_not_relaxed():
    params = {"confidence_call_threshold": 0.75}
    assert deploy_params_for_eval(params, {"eval_relaxed_gating": False}) == params


def test_resolve_deploy_ok_mini_and_disabled():
    cfg = {"enabled": False}
    assert resolve_deploy_ok(mini_ok=True, val_accuracy=0.4, val_brier=0.9, gate_cfg=cfg) is True
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.4, val_brier=0.9, gate_cfg=cfg) is True
