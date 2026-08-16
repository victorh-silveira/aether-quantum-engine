from src.domain.risk.risk_policy import load_risk_policy, validate_engine_risk_config


def test_load_risk_policy_reads_nested_defaults():
    policy = load_risk_policy(
        {
            "orchestrator": {"execution": {"mandatory_trade_each_cycle": False, "require_meta_for_execution": True}},
            "risk_management": {"kelly": {"max_stake_pct": 0.02, "max_bankroll_stake_fraction": 0.02}},
            "deep_learning": {"deploy_gate": {"enabled": True}},
            "infra": {},
        }
    )
    assert policy.mandatory_trade_each_cycle is False
    assert policy.require_meta_for_execution is True
    assert policy.max_stake_pct == 0.02
    assert policy.deploy_gate_enabled is True


def test_validate_engine_risk_config_flags_loose_caps():
    issues = validate_engine_risk_config(
        {
            "orchestrator": {"execution": {"mandatory_trade_each_cycle": True}},
            "risk_management": {"kelly": {"max_stake_pct": 1.0, "max_bankroll_stake_fraction": 1.0}},
            "deep_learning": {"deploy_gate": {"enabled": False}},
        }
    )
    assert any("max_stake_pct" in item for item in issues)
    assert any("deploy_gate.enabled" in item for item in issues)


def test_validate_engine_risk_config_flags_fraction_below_max_pct():
    issues = validate_engine_risk_config(
        {
            "orchestrator": {"execution": {"mandatory_trade_each_cycle": False}},
            "risk_management": {
                "kelly": {
                    "max_stake_pct": 0.04,
                    "max_bankroll_stake_fraction": 0.02,
                    "recovery_min_trade_score": 0.30,
                }
            },
            "deep_learning": {"deploy_gate": {"enabled": True}},
        }
    )
    assert any("max_bankroll_stake_fraction < max_stake_pct" in item for item in issues)
    assert any("recovery_min_trade_score suspeito" in item for item in issues)
