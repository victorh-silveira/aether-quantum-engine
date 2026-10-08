import pytest

from src.infrastructure.inference.loss_classifier_types import BUFFER_N_ABSENT, parse_loss_predict_response


def test_parse_loss_predict_response_full_payload():
    parsed = parse_loss_predict_response(
        {
            "p_loss": 0.91,
            "veto": True,
            "auto_learn_applied": True,
            "model_version": "v3",
            "n_train": 42,
            "veto_ready": True,
            "bootstrap": True,
            "collapsed": True,
            "buffer_n": 4,
            "bootstrap_exit_n": 4,
        }
    )
    assert parsed["p_loss"] == pytest.approx(0.91)
    assert parsed["veto"] is True
    assert parsed["auto_learn_applied"] is True
    assert parsed["model_version"] == "v3"
    assert parsed["n_train"] == 42
    assert parsed["veto_ready"] is True
    assert parsed["bootstrap"] is True
    assert parsed["collapsed"] is True
    assert parsed["buffer_n"] == 4
    assert parsed["bootstrap_exit_n"] == 4


def test_parse_loss_predict_response_defaults():
    parsed = parse_loss_predict_response({})
    assert parsed["p_loss"] == pytest.approx(0.5)
    assert parsed["veto"] is False
    assert parsed["auto_learn_applied"] is False
    assert parsed["model_version"] == "none"
    assert parsed["n_train"] == 0
    assert parsed["veto_ready"] is False
    assert parsed["bootstrap"] is False
    assert parsed["collapsed"] is False
    assert parsed["buffer_n"] == BUFFER_N_ABSENT
    assert parsed["bootstrap_exit_n"] == 4


def test_parse_loss_predict_response_explicit_zero_buffer():
    parsed = parse_loss_predict_response({"buffer_n": 0, "bootstrap_exit_n": 4})
    assert parsed["buffer_n"] == 0
    assert parsed["bootstrap_exit_n"] == 4


def test_parse_loss_predict_response_rejects_non_object():
    with pytest.raises(TypeError, match="loss response must be object"):
        parse_loss_predict_response([])
