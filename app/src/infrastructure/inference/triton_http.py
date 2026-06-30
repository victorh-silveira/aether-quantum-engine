"""HTTP seguro (esquemas http/https) para NVIDIA Triton Inference Server."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

from src.infrastructure.api.deriv_http import read_http_response


def triton_http_base_url(raw_url: str) -> str:
    """Normaliza URL HTTP base do Triton."""
    http_url = str(raw_url).rstrip("/")
    if not http_url.startswith("http"):
        http_url = f"http://{http_url}"
    return http_url


def get_triton_model_metadata(http_base: str, model_name: str) -> dict[str, Any]:
    """Busca metadados do modelo via GET /v2/models/{name}."""
    url = f"{triton_http_base_url(http_base)}/v2/models/{model_name}"
    req = urllib.request.Request(url, method="GET")
    body = read_http_response(req, timeout=30).decode("utf-8")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Triton metadata invalida para {model_name}: {type(parsed).__name__}")
    if parsed.get("error"):
        raise RuntimeError(f"Triton metadata erro para {model_name}: {parsed['error']}")
    return parsed


def post_triton_repository_reload(http_url: str) -> list[dict[str, str]]:
    """Solicita rescan do model repository via API HTTP do Triton."""
    payload = json.dumps({"action": "reload"}).encode("utf-8")
    base = triton_http_base_url(http_url)
    req = urllib.request.Request(
        f"{base}/v2/repository/index",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    body = read_http_response(req, timeout=60).decode("utf-8")
    parsed = json.loads(body)
    if isinstance(parsed, list):
        return parsed
    return []
