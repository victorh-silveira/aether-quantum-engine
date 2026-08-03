"""Contrato da stack Docker hibrida e smoke opt-in em localhost."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request

import pytest

from aether_paths import repo_path


def _compose_text() -> str:
    return repo_path("infra", "docker", "docker-compose.yml").read_text(encoding="utf-8")


def _gpu_overlay_text() -> str:
    return repo_path("infra", "docker", "docker-compose.gpu.yml").read_text(encoding="utf-8")


def test_compose_base_has_hardening_and_localhost_binds():
    text = _compose_text()
    assert "restart: unless-stopped" in text
    assert 'max-size: "10m"' in text
    assert "127.0.0.1:6379:6379" in text
    assert "127.0.0.1:5432:5432" in text
    assert "127.0.0.1:9000:9000" in text
    assert "127.0.0.1:8000:8000" in text
    assert "127.0.0.1:8005:8000" in text
    assert "mem_limit: 256m" in text
    assert "no-new-privileges:true" in text
    assert "gpus:" not in text
    assert "profiles: [gpu, cpu]" in text
    assert "profiles: [core, gpu, cpu]" in text


def test_compose_gpu_overlay_declares_nvidia():
    text = _gpu_overlay_text()
    assert "gpus: all" in text
    assert "nvidia" in text
    assert "aether-triton" in text


def test_timescale_orphan_conf_removed():
    assert not repo_path("infra", "docker", "timescaledb-aether-io.conf").is_file()


def test_meta_dockerfile_non_root_and_healthcheck():
    text = repo_path("infra", "docker", "meta-classifier", "Dockerfile").read_text(encoding="utf-8")
    assert "USER aether" in text
    assert "HEALTHCHECK" in text
    assert repo_path("infra", "docker", "meta-classifier", ".dockerignore").is_file()


def test_compose_lib_and_env_example_document_gpu_knobs():
    lib = repo_path("infra", "docker", "compose-lib.sh").read_text(encoding="utf-8")
    env = repo_path(".env.example").read_text(encoding="utf-8")
    assert "DOCKER_GPU" in lib
    assert "docker-compose.gpu.yml" in lib
    assert "AETHER_TRITON_HTTP" in env
    assert "AETHER_META_CLASSIFIER_HTTP" in env
    assert "DOCKER_PROFILES" in env


def _tcp_open(host: str, port: int, timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.getcode()) == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


@pytest.mark.docker
def test_live_core_stack_localhost():
    if not _tcp_open("127.0.0.1", 6379):
        pytest.skip("Redis localhost:6379 indisponivel")
    if not _tcp_open("127.0.0.1", 5432):
        pytest.skip("Timescale localhost:5432 indisponivel")
    if not _http_ok("http://127.0.0.1:9000/minio/health/live"):
        pytest.skip("MinIO localhost:9000 indisponivel")


@pytest.mark.docker
def test_live_triton_and_meta_when_up():
    triton_up = _http_ok("http://127.0.0.1:8000/v2/health/live")
    meta_up = False
    meta_ready = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8005/health", timeout=1.0) as response:
            payload = json.loads(response.read().decode())
            meta_up = int(response.getcode()) == 200
            meta_ready = bool(payload.get("ready"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass
    if not triton_up and not meta_up:
        pytest.skip("Triton/Meta nao estao no ar")
    if triton_up:
        assert triton_up is True
    if meta_up:
        assert meta_ready is True
