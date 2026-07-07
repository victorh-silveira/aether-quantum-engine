"""HTTP seguro (esquemas http/https) para NVIDIA Triton Inference Server."""

from __future__ import annotations

import json
import time
import urllib.error
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


def fetch_triton_health_ready(http_url: str) -> None:
    """Valida endpoint HTTP /v2/health/ready do Triton."""
    base = triton_http_base_url(http_url)
    req = urllib.request.Request(f"{base}/v2/health/ready", method="GET")
    body = read_http_response(req, timeout=2.0).decode("utf-8").strip()
    if body and body.lower() not in ("", "ok"):
        parsed = json.loads(body) if body.startswith("{") else {}
        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(f"Triton health/ready: {parsed['error']}")


def triton_model_ready(http_base: str, model_name: str) -> bool:
    """Retorna True quando GET /v2/models/{name}/ready responde 200."""
    base = triton_http_base_url(http_base)
    quoted = urllib.parse.quote(str(model_name), safe="")
    req = urllib.request.Request(f"{base}/v2/models/{quoted}/ready", method="GET")
    try:
        read_http_response(req, timeout=2.0)
        return True
    except urllib.error.HTTPError as exc:
        if exc.code in (400, 404, 503):
            return False
        raise


def wait_triton_models_ready(
    http_base: str,
    model_names: list[str],
    *,
    timeout_seconds: float = 25.0,
    poll_interval_seconds: float = 0.5,
) -> bool:
    """Aguarda modelos ficarem prontos apos sync no repositorio em modo poll."""
    pending = {str(name) for name in model_names if str(name)}
    if not pending:
        return True
    deadline = time.monotonic() + max(0.5, float(timeout_seconds))
    poll = max(0.1, float(poll_interval_seconds))
    while pending and time.monotonic() < deadline:
        for name in list(pending):
            if triton_model_ready(http_base, name):
                pending.discard(name)
        if not pending:
            return True
        time.sleep(poll)
    return False


def post_triton_repository_reload(http_url: str) -> list[dict[str, str]]:
    """Lista o indice do model repository via POST /v2/repository/index."""
    payload = json.dumps({}).encode("utf-8")
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
