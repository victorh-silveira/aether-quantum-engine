from src.application.services.market_audit_log import format_gates_audit_line


def test_format_gates_audit_line_loss_clf_only():
    line = format_gates_audit_line(
        {
            "loss_clf_p_loss": 0.91,
            "loss_clf_hard": True,
            "loss_clf_hard_p_loss_floor": 0.9,
            "loss_clf_n_train": 32,
            "loss_clf_auto_learn": True,
            "gate_reason": "loss_clf",
            "gate_verdict": "HARD_SKIP",
        }
    )
    assert line.startswith("[GATES] || LOSS_CLF: HARD")
    assert "skip=loss_clf" in line
    assert "verdict=HARD_SKIP" in line
    assert "FUSION" not in line
    assert "MICRO" not in line
    assert "NEG_EDGE" not in line


def test_format_gates_audit_line_ok_and_off():
    ok = format_gates_audit_line(
        {
            "loss_clf_p_loss": 0.4,
            "loss_clf_n_train": 8,
            "loss_clf_auto_learn": False,
            "loss_clf_veto_ready": True,
            "loss_clf_model_version": "v1",
            "gate_verdict": "ALLOW",
        }
    )
    assert "LOSS_CLF: OK" in ok
    assert "skip=-" in ok
    off = format_gates_audit_line({})
    assert "LOSS_CLF: OFF" in off
