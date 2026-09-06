from src.application.services.execution_micro_protect import apply_micro_protect_gates
from tests.unit.application.fixtures.micro_protect_session_cases import MICRO_PROTECT_SESSION_CASES


def test_micro_protect_session_fixtures():
    for case in MICRO_PROTECT_SESSION_CASES:
        metrics = dict(case["metrics"])
        hard = apply_micro_protect_gates(metrics)
        assert hard is case["expect_hard"], case["name"]
        if case["expect_hard"]:
            assert metrics["gate_reason"] == case["expected_reason"], case["name"]
        else:
            assert metrics.get("gate_reason") != "micro_discord"
            assert metrics.get("gate_reason") != "chop_loss_risk"
            assert metrics.get("gate_reason") != "soft_confirm_weak"
            assert metrics.get("execution_candidate_ready") is True
