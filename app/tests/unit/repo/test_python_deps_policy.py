"""Politica anti-redundancia de requirements Python."""

from __future__ import annotations

import re

from aether_paths import repo_path


_REQ_LINE = re.compile(r"^\s*([A-Za-z0-9_.\-]+)")
_EXTRA_DF = frozenset(
    {
        "modin",
        "dask",
        "cudf",
        "vaex",
        "datatable",
        "pyarrow-dataset",
    }
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


def test_requirements_dev_no_coverage_with_pytest_cov():
    names = _package_names(repo_path("app", "requirements-dev.txt"))
    if "pytest-cov" in names:
        assert "coverage" not in names


def test_requirements_no_third_dataframe_lib():
    paths = [
        repo_path("app", "requirements.txt"),
        repo_path("app", "requirements-dev.txt"),
        repo_path("infra", "docker", "meta-classifier", "requirements.txt"),
        repo_path("infra", "docker", "loss-classifier", "requirements.txt"),
    ]
    allowed = {"pandas", "polars"}
    for path in paths:
        names = _package_names(path)
        forbidden = (names & _EXTRA_DF) - allowed
        assert not forbidden, f"{path}: libs DF extras {sorted(forbidden)}"
