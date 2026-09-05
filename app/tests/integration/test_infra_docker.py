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


def test_compose_base_has_hardening_and_localhost_binds():
    text = _compose_text()
    assert "restart: unless-stopped" in text
    assert 'max-size: "10m"' in text
    assert "127.0.0.1:6379:6379" in text
    assert "127.0.0.1:5432:5432" in text
    assert "127.0.0.1:9000:9000" in text
    assert "127.0.0.1:8005:8000" in text
    assert "127.0.0.1:8006:8000" in text
    assert "mem_limit: 256m" in text
    assert "mem_swappiness: 0" in text
    assert "no-new-privileges:true" in text
    assert "cap_drop:" in text
    assert "- ALL" in text
    assert "read_only: true" in text
    assert "tmpfs:" in text
    assert "nofile:" in text
    assert "gpus:" not in text
    assert "aether-triton" not in text
    assert "profiles: [core]" in text
    assert "profiles: [ml]" in text
    assert "condition: service_healthy" in text
    assert "condition: service_completed_successfully" in text
    assert "minio-init:" in text
    assert 'aether.oneshot: "true"' in text
    assert 'OMP_NUM_THREADS: "2"' in text
    assert 'MKL_NUM_THREADS: "2"' in text
    assert '["server", "/data", "--console-address", ":9001"]' in text


def test_compose_gpu_overlay_removed():
    assert not repo_path("infra", "docker", "docker-compose.gpu.yml").is_file()


def test_timescale_orphan_conf_removed():
    assert not repo_path("infra", "docker", "timescaledb-aether-io.conf").is_file()


def test_timescale_sql_chunk_and_crags():
    init_sql = repo_path("infra", "docker", "003_init-timescale.sql").read_text(encoding="utf-8")
    crags = repo_path("infra", "docker", "005_timescale_crags.sql").read_text(encoding="utf-8")
    assert "chunk_time_interval => INTERVAL '1 day'" in init_sql
    assert "candle_m5" in crags
    assert "timescaledb.continuous" in crags
    assert "time_bucket(INTERVAL '5 minutes'" in crags
    lifecycle = repo_path("infra", "docker", "timescale-lifecycle.sh").read_text(encoding="utf-8")
    assert "005_timescale_crags.sql" in lifecycle


def test_minio_init_script_bucket_ilm():
    script = repo_path("infra", "docker", "minio-init.sh").read_text(encoding="utf-8")
    assert "dl-models" in script
    assert "optuna/" in script
    assert "mc ilm import" in script


def test_docker_hydrate_uses_1hz75v_m5_d1():
    script = repo_path("infra", "docker", "docker-hydrate.sh").read_text(encoding="utf-8")
    assert "1HZ75V" in script
    assert "300" in script
    assert "86400" in script
    assert "R_10" not in script
    assert "7200" not in script
    assert "granularity=60" not in script


def test_timescale_lifecycle_ohlc_segmentby_includes_granularity():
    sql = repo_path("infra", "docker", "004_timescale-lifecycle.sql").read_text(encoding="utf-8")
    assert "symbol,granularity" in sql
    assert "time DESC, epoch DESC" in sql


def test_aether_io_tune_reloadable_only():
    sql = repo_path("infra", "docker", "002_aether-io-tune.sql").read_text(encoding="utf-8")
    assert "shared_buffers" not in sql
    assert "work_mem" in sql
    assert "pg_reload_conf" in sql


def test_makefile_docker_logs_defaults_running_services():
    text = repo_path("Makefile").read_text(encoding="utf-8")
    assert "DOCKER_LOGS_TAIL ?= 200" in text
    assert "DOCKER_LOGS_SERVICES ?= redis timescaledb minio aether-meta-classifier aether-loss-classifier" in text
    assert "minio-init" not in text.split("DOCKER_LOGS_SERVICES")[1].split("\n")[0]


def test_meta_dockerfile_non_root_and_healthcheck():
    text = repo_path("infra", "docker", "meta-classifier", "Dockerfile").read_text(encoding="utf-8")
    assert text.count("FROM python:3.13-slim") >= 2
    assert "/opt/venv" in text
    assert "USER aether" in text
    assert "HEALTHCHECK" in text
    assert "tini" in text
    assert 'ENTRYPOINT ["/usr/bin/tini", "--"]' in text
    assert "MKL_NUM_THREADS=2" in text
    assert repo_path("infra", "docker", "meta-classifier", ".dockerignore").is_file()


def test_loss_dockerfile_multi_stage():
    text = repo_path("infra", "docker", "loss-classifier", "Dockerfile").read_text(encoding="utf-8")
    assert text.count("FROM python:3.13-slim") >= 2
    assert "/opt/venv" in text
    assert "LOSS_BOOTSTRAP_EXIT_N=8" in text


def test_compose_lib_and_env_example_document_ml_knobs():
    lib = repo_path("infra", "docker", "compose-lib.sh").read_text(encoding="utf-8")
    env = repo_path(".env.example").read_text(encoding="utf-8")
    assert "DOCKER_PROFILES" in lib
    assert "docker-compose.gpu.yml" not in lib
    assert "AETHER_TRITON_HTTP" not in env
    assert "AETHER_META_CLASSIFIER_HTTP" in env
    assert "AETHER_LOSS_CLASSIFIER_HTTP" in env
    assert "DOCKER_PROFILES=core,ml" in env


def test_compose_loss_classifier_env_ssot():
    text = _compose_text()
    assert 'LOSS_READY_N: "8"' in text
    assert 'LOSS_BOOTSTRAP_EXIT_N: "8"' in text
    assert 'LOSS_MIN_WIN_FOR_LOSS_RETRAIN: "1"' in text
    dockerfile = repo_path("infra", "docker", "loss-classifier", "Dockerfile").read_text(encoding="utf-8")
    assert "LOSS_READY_N=8" in dockerfile
    assert "LOSS_BOOTSTRAP_EXIT_N=8" in dockerfile
    assert "LOSS_MIN_WIN_FOR_LOSS_RETRAIN=1" in dockerfile


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
def test_live_meta_when_up():
    meta_up = False
    meta_ready = False
    try:
        with urllib.request.urlopen("http://127.0.0.1:8005/health", timeout=1.0) as response:
            payload = json.loads(response.read().decode())
            meta_up = int(response.getcode()) == 200
            meta_ready = bool(payload.get("ready")) or "model_loaded" in payload
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        pass
    if not meta_up:
        pytest.skip("Meta classifier nao esta no ar")
    assert meta_ready is True
