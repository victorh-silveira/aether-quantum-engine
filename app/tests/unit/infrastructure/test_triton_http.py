import json
import urllib.error
from unittest.mock import patch

import pytest

from src.infrastructure.inference.triton_http import (
    fetch_triton_health_ready,
    get_triton_model_metadata,
    post_triton_repository_reload,
    triton_http_base_url,
    triton_model_ready,
    wait_triton_models_ready,
)


def test_triton_http_base_url_adds_scheme():
    assert triton_http_base_url("localhost:8000") == "http://localhost:8000"
    assert triton_http_base_url("http://triton:8000/") == "http://triton:8000"


def test_get_triton_model_metadata_success():
    payload = {"name": "RDBEAR", "inputs": [{"shape": [-1, 48, 34]}]}
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=json.dumps(payload).encode("utf-8"),
    ):
        out = get_triton_model_metadata("http://localhost:8000", "RDBEAR")
    assert out["name"] == "RDBEAR"


def test_get_triton_model_metadata_error_field():
    with (
        patch(
            "src.infrastructure.inference.triton_http.read_http_response",
            return_value=json.dumps({"error": "not found"}).encode("utf-8"),
        ),
        pytest.raises(RuntimeError, match="not found"),
    ):
        get_triton_model_metadata("http://localhost:8000", "RDBEAR")


def test_get_triton_model_metadata_invalid_type():
    with (
        patch(
            "src.infrastructure.inference.triton_http.read_http_response",
            return_value=b"[]",
        ),
        pytest.raises(RuntimeError, match="invalida"),
    ):
        get_triton_model_metadata("http://localhost:8000", "RDBEAR")


def test_triton_model_ready_true_and_false():
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=b"",
    ):
        assert triton_model_ready("http://localhost:8000", "RDBEAR") is True
    for code in (400, 404, 503):
        with patch(
            "src.infrastructure.inference.triton_http.read_http_response",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:8000/v2/models/RDBEAR/ready",
                code=code,
                msg="Not Ready",
                hdrs=None,
                fp=None,
            ),
        ):
            assert triton_model_ready("http://localhost:8000", "RDBEAR") is False


def test_triton_model_ready_reraises_unexpected_http_error():
    with (
        patch(
            "src.infrastructure.inference.triton_http.read_http_response",
            side_effect=urllib.error.HTTPError(
                url="http://localhost:8000/v2/models/RDBEAR/ready",
                code=500,
                msg="Server Error",
                hdrs=None,
                fp=None,
            ),
        ),
        pytest.raises(urllib.error.HTTPError),
    ):
        triton_model_ready("http://localhost:8000", "RDBEAR")


def test_wait_triton_models_ready_empty_list():
    assert wait_triton_models_ready("http://localhost:8000", []) is True


def test_wait_triton_models_ready_eventually():
    state = {"n": 0}

    def _ready(_base: str, name: str) -> bool:
        state["n"] += 1
        return state["n"] >= 2

    with patch("src.infrastructure.inference.triton_http.triton_model_ready", side_effect=_ready):
        ok = wait_triton_models_ready(
            "http://localhost:8000",
            ["RDBEAR"],
            timeout_seconds=2.0,
            poll_interval_seconds=0.01,
        )
    assert ok is True


def test_wait_triton_models_ready_timeout():
    with patch("src.infrastructure.inference.triton_http.triton_model_ready", return_value=False):
        ok = wait_triton_models_ready(
            "http://localhost:8000",
            ["RDBEAR"],
            timeout_seconds=0.2,
            poll_interval_seconds=0.05,
        )
    assert ok is False


def test_post_triton_repository_reload_list():
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=json.dumps([{"name": "RDBEAR"}]).encode("utf-8"),
    ):
        out = post_triton_repository_reload("localhost:8000")
    assert out == [{"name": "RDBEAR"}]


def test_post_triton_repository_reload_non_list():
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=json.dumps({"error": "x"}).encode("utf-8"),
    ):
        out = post_triton_repository_reload("http://localhost:8000")
    assert out == []


def test_fetch_triton_health_ready_ok():
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=b"ok",
    ):
        fetch_triton_health_ready("http://localhost:8000")


def test_fetch_triton_health_ready_error_json():
    with (
        patch(
            "src.infrastructure.inference.triton_http.read_http_response",
            return_value=b'{"error":"down"}',
        ),
        pytest.raises(RuntimeError, match="health/ready"),
    ):
        fetch_triton_health_ready("http://localhost:8000")
