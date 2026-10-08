"""Politica anti-redundancia de requirements Python."""

from __future__ import annotations

import re
from pathlib import Path

from aether_paths import repo_path


_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")
_PANDAS_IMPORT = re.compile(r"^\s*(import\s+pandas\b|from\s+pandas\b)")
_EXTRA_DF = frozenset(
    {
        "modin",
        "dask",
        "cudf",
        "vaex",
        "datatable",
        "pyarrow-dataset",
        "pandas",
    }
)
_SCAN_ROOTS = ("app", "infra/docker")
_HOST_PIN_MARKERS = (
    "websockets==16.0",
    "httpx==0.28.1",
    "numpy==2.4.6",
    "polars==1.23.0",
    "torch==2.10.0",
)


def _package_names(path) -> set[str]:
    names: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = _REQ_LINE.match(line)
        if not match:
            continue
        names.add(match.group(1).split("[", 1)[0].lower().replace("_", "-"))
    return names


def _requirement_files() -> list[Path]:
    return [
        repo_path("app", "requirements.txt"),
        repo_path("app", "requirements-dev.txt"),
        repo_path("infra", "docker", "meta-classifier", "requirements.txt"),
        repo_path("infra", "docker", "loss-classifier", "requirements.txt"),
    ]


def test_requirements_dev_no_coverage_with_pytest_cov():
    names = _package_names(repo_path("app", "requirements-dev.txt"))
    if "pytest-cov" in names:
        assert "coverage" not in names


def test_requirements_forbid_pandas_and_extra_dataframe_libs():
    for path in _requirement_files():
        names = _package_names(path)
        assert "pandas" not in names, f"{path}: pandas proibido"
        forbidden = names & (_EXTRA_DF - {"pandas"})
        assert not forbidden, f"{path}: libs DF extras {sorted(forbidden)}"
        if path.name == "requirements.txt" and "app" in path.parts:
            assert "polars" in names


def test_host_requirements_keep_senior_pins():
    text = repo_path("app", "requirements.txt").read_text(encoding="utf-8")
    for marker in _HOST_PIN_MARKERS:
        assert marker in text, f"pin ausente: {marker}"


def test_first_party_has_no_pandas_import():
    offenders: list[str] = []
    root = repo_path()
    for rel in _SCAN_ROOTS:
        base = root / rel
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for line in text.splitlines():
                if _PANDAS_IMPORT.match(line):
                    offenders.append(str(path.relative_to(root)).replace("\\", "/"))
                    break
    assert not offenders, f"import pandas proibido: {offenders}"


def test_minio_model_store_offloads_sync_sdk():
    text = repo_path("app", "src", "infrastructure", "storage", "minio_model_store.py").read_text(encoding="utf-8")
    assert "asyncio.to_thread" in text


def test_meta_http_client_is_persistent_builder():
    text = repo_path("app", "src", "infrastructure", "inference", "meta_classifier_client.py").read_text(
        encoding="utf-8"
    )
    assert "def build_persistent_http_client" in text
    assert "httpx.Limits" in text
