import json
from unittest.mock import patch

import pytest

from src.infrastructure.inference.triton_http import (
    get_triton_model_metadata,
    post_triton_repository_reload,
    triton_http_base_url,
)


def test_triton_http_base_url_adds_scheme():
    assert triton_http_base_url("localhost:8000") == "http://localhost:8000"
    assert triton_http_base_url("http://triton:8000/") == "http://triton:8000"


def test_get_triton_model_metadata_success():
    payload = {"name": "R_10", "inputs": [{"shape": [-1, 48, 34]}]}
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=json.dumps(payload).encode("utf-8"),
    ):
        out = get_triton_model_metadata("http://localhost:8000", "R_10")
    assert out["name"] == "R_10"


def test_get_triton_model_metadata_error_field():
    with (
        patch(
            "src.infrastructure.inference.triton_http.read_http_response",
            return_value=json.dumps({"error": "not found"}).encode("utf-8"),
        ),
        pytest.raises(RuntimeError, match="not found"),
    ):
        get_triton_model_metadata("http://localhost:8000", "R_10")


def test_get_triton_model_metadata_invalid_type():
    with (
        patch(
            "src.infrastructure.inference.triton_http.read_http_response",
            return_value=b"[]",
        ),
        pytest.raises(RuntimeError, match="invalida"),
    ):
        get_triton_model_metadata("http://localhost:8000", "R_10")


def test_post_triton_repository_reload_list():
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=json.dumps([{"name": "R_10"}]).encode("utf-8"),
    ):
        out = post_triton_repository_reload("localhost:8000")
    assert out == [{"name": "R_10"}]


def test_post_triton_repository_reload_non_list():
    with patch(
        "src.infrastructure.inference.triton_http.read_http_response",
        return_value=json.dumps({"error": "x"}).encode("utf-8"),
    ):
        out = post_triton_repository_reload("http://localhost:8000")
    assert out == []
