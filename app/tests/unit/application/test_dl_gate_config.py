from src.application.services.deep_learning.dl_gate_config import (
    deploy_params_for_eval,
    parse_deploy_gate_config,
    resolve_deploy_ok,
)


def test_resolve_deploy_ok_soft_fallback():
    cfg = parse_deploy_gate_config(
        {
            "deploy_gate": {
                "enabled": True,
                "force_ok": False,
                "soft_min_val_accuracy": 0.50,
                "soft_max_brier": 0.32,
            }
        }
    )
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.52, val_brier=0.25, gate_cfg=cfg) is True
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.48, val_brier=0.25, gate_cfg=cfg) is False


def test_resolve_deploy_ok_force_ok_requires_acc_floor():
    cfg = parse_deploy_gate_config({"deploy_gate": {"enabled": True, "force_ok": True}})
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.55, val_brier=0.30, gate_cfg=cfg) is True
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.40, val_brier=0.30, gate_cfg=cfg) is False


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


def test_resolve_deploy_ok_mini_cannot_bypass_acc_floor():
    cfg = {
        "enabled": True,
        "force_ok": False,
        "soft_min_val_accuracy": 0.53,
        "soft_max_brier": 0.24,
    }
    assert resolve_deploy_ok(mini_ok=True, val_accuracy=0.52, val_brier=0.20, gate_cfg=cfg) is False
    assert resolve_deploy_ok(mini_ok=True, val_accuracy=0.54, val_brier=0.20, gate_cfg=cfg) is True


def test_resolve_deploy_ok_disabled_still_respects_acc():
    cfg = {"enabled": False, "soft_min_val_accuracy": 0.53, "soft_max_brier": 0.32}
    assert resolve_deploy_ok(mini_ok=True, val_accuracy=0.40, val_brier=0.20, gate_cfg=cfg) is False
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.55, val_brier=0.20, gate_cfg=cfg) is True


def test_resolve_deploy_ok_soft_fallback_near_coin_flip_brier():
    cfg = parse_deploy_gate_config(
        {
            "deploy_gate": {
                "enabled": True,
                "force_ok": False,
                "soft_min_val_accuracy": 0.53,
                "soft_max_brier": 0.26,
            }
        }
    )
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.566, val_brier=0.250, gate_cfg=cfg) is True
    assert resolve_deploy_ok(mini_ok=False, val_accuracy=0.566, val_brier=0.270, gate_cfg=cfg) is False


def test_describe_deploy_block_brier():
    from src.application.services.deep_learning.dl_gate_config import describe_deploy_block

    cfg = {"soft_min_val_accuracy": 0.53, "soft_max_brier": 0.26, "enabled": True}
    msg = describe_deploy_block(mini_ok=False, val_accuracy=0.56, val_brier=0.27, gate_cfg=cfg)
    assert "val_brier" in msg


def test_resolve_deploy_ok_rejects_majority_collapse():
    cfg = {
        "enabled": True,
        "force_ok": False,
        "soft_min_val_accuracy": 0.53,
        "soft_max_brier": 0.32,
        "reject_majority_collapse": True,
        "max_label_call_frac_bias": 0.20,
        "min_minority_recall": 0.25,
    }
    assert (
        resolve_deploy_ok(
            mini_ok=True,
            val_accuracy=0.60,
            val_brier=0.20,
            gate_cfg=cfg,
            label_call_frac=0.80,
            minority_recall=0.10,
        )
        is False
    )
    assert (
        resolve_deploy_ok(
            mini_ok=True,
            val_accuracy=0.60,
            val_brier=0.20,
            gate_cfg=cfg,
            label_call_frac=0.55,
            minority_recall=0.10,
        )
        is True
    )


def test_describe_deploy_block_majority_collapse():
    from src.application.services.deep_learning.dl_gate_config import describe_deploy_block

    cfg = {
        "soft_min_val_accuracy": 0.53,
        "soft_max_brier": 0.26,
        "enabled": True,
        "reject_majority_collapse": True,
        "max_label_call_frac_bias": 0.20,
        "min_minority_recall": 0.25,
    }
    msg = describe_deploy_block(
        mini_ok=True,
        val_accuracy=0.60,
        val_brier=0.20,
        gate_cfg=cfg,
        label_call_frac=0.85,
        minority_recall=0.05,
    )
    assert "majority_collapse" in msg


def test_describe_deploy_block_remaining_branches():
    from src.application.services.deep_learning.dl_gate_config import describe_deploy_block

    cfg = {"soft_min_val_accuracy": 0.53, "soft_max_brier": 0.26, "enabled": True}
    assert "soft_min" in describe_deploy_block(mini_ok=False, val_accuracy=0.40, val_brier=0.1, gate_cfg=cfg)
    assert "inesperado" in describe_deploy_block(mini_ok=True, val_accuracy=0.60, val_brier=0.1, gate_cfg=cfg)
    assert "sem motivo" in describe_deploy_block(mini_ok=False, val_accuracy=0.60, val_brier=0.1, gate_cfg=cfg)
