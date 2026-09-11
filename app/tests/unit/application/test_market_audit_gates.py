from src.application.services.market_audit_log import format_gates_audit_line


def test_format_gates_audit_line_loss_clf_flip():
    line = format_gates_audit_line(
        {
            "loss_clf_p_loss": 0.91,
            "loss_clf_p_eff": 0.91,
            "loss_clf_flip": True,
            "loss_clf_flip_floor": 0.55,
            "loss_clf_hard_p_loss_floor": 0.55,
            "loss_clf_n_train": 32,
            "loss_clf_auto_learn": True,
            "gate_verdict": "ALLOW",
        }
    )
    assert line.startswith("[GATES] || LOSS_CLF: FLIP")
    assert "skip=-" in line
    assert "verdict=ALLOW" in line
    assert "floor=0.55" in line
    assert " pe=" not in line
    assert "HARD" not in line
    assert "FUSION" not in line
    assert "MICRO" not in line


def test_format_gates_audit_line_shows_pe_when_differs():
    line = format_gates_audit_line(
        {
            "loss_clf_p_loss": 0.71,
            "loss_clf_p_eff": 0.5735,
            "loss_clf_flip": True,
            "loss_clf_flip_floor": 0.55,
            "loss_clf_hard_p_loss_floor": 0.55,
            "loss_clf_n_train": 12,
            "loss_clf_auto_learn": True,
        }
    )
    assert "p=0.71000" in line
    assert "pe=0.57350" in line
    assert "floor=0.55" in line


def test_format_gates_audit_line_ok_and_off():
    ok = format_gates_audit_line(
        {
            "loss_clf_p_loss": 0.4,
            "loss_clf_p_eff": 0.4,
            "loss_clf_n_train": 8,
            "loss_clf_auto_learn": False,
            "loss_clf_veto_ready": True,
            "loss_clf_model_version": "v1",
            "gate_verdict": "ALLOW",
        }
    )
    assert "LOSS_CLF: OK" in ok
    assert "skip=-" in ok
    assert "blocked=" not in ok
    off = format_gates_audit_line({})
    assert "LOSS_CLF: OFF" in off


def test_format_gates_audit_line_ok_shows_blocked_bootstrap():
    line = format_gates_audit_line(
        {
            "loss_clf_p_loss": 0.64431,
            "loss_clf_p_eff": 0.64431,
            "loss_clf_n_train": 64,
            "loss_clf_auto_learn": False,
            "loss_clf_veto_ready": True,
            "loss_clf_model_version": "loss_bootstrap_live64",
            "loss_clf_flip_blocked": "bootstrap",
            "loss_clf_buffer_n": 2,
            "loss_clf_bootstrap_exit_n": 4,
        }
    )
    assert "LOSS_CLF: OK" in line
    assert "auto=0" in line
    assert "blocked=bootstrap" in line
    assert "boot=2/4" in line
    assert "ready_seed=1" in line
    assert "ready=1" not in line.replace("ready_seed=1", "")
    assert "ver=loss_bootstrap_live64" in line
    assert "skip=-" in line
