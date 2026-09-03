import subprocess
import sys
from pathlib import Path

import pytest

from scripts.operations import clean_workspace as cw
from scripts.operations.qa.common import AREAS, STAGES, is_ci, posix_for_tool, require_tool, run_cmd
from scripts.operations.qa.dispatch import run_area_stage, run_config_text
from scripts.operations.qa.docker import run_docker
from scripts.operations.qa.json_area import run_json
from scripts.operations.qa.shell import run_shell
from scripts.operations.qa.yaml_area import run_yaml


def test_posix_for_tool_windows_drive():
    converted = posix_for_tool(Path("C:/tmp/ok.sh"))
    if converted.startswith("/mnt/"):
        assert converted.startswith("/mnt/c/")
        assert converted.endswith("ok.sh")


def test_matrix_areas_and_stages():
    assert AREAS == ("python", "docker", "shell")
    assert "lint" in STAGES and "validate" in STAGES and "build" in STAGES


def test_is_ci_reads_env(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    assert is_ci() is False
    monkeypatch.setenv("CI", "true")
    assert is_ci() is True


def test_json_lint_validate_test(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.json").write_text('{"symbols": ["1HZ75V"], "deep_learning": {}}', encoding="utf-8")
    (config / "python.json").write_text('{"conda_env": "deriv-api"}', encoding="utf-8")
    run_json("lint", tmp_path)
    run_json("validate", tmp_path)
    run_json("test", tmp_path)
    run_json("security", tmp_path)
    run_json("build", tmp_path)
    run_config_text("lint", tmp_path)
    run_config_text("lint", tmp_path, kind="json")
    run_config_text("lint", tmp_path, kind="yaml")


def test_config_text_unknown_kind(tmp_path: Path):
    with pytest.raises(ValueError, match="config-text desconhecido"):
        run_config_text("lint", tmp_path, kind="xml")


def test_json_validate_requires_ssot_keys(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="symbols"):
        run_json("validate", tmp_path)


def test_json_test_requires_conda_env(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.json").write_text('{"symbols": ["x"], "deep_learning": {}}', encoding="utf-8")
    (config / "python.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="conda_env"):
        run_json("test", tmp_path)


def test_yaml_lint_rejects_tabs(tmp_path: Path):
    compose = tmp_path / "infra" / "docker"
    compose.mkdir(parents=True)
    (compose / "docker-compose.yml").write_text("services:\n\tredis: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tab"):
        run_yaml("lint", tmp_path)


def test_yaml_lint_rejects_empty(tmp_path: Path):
    compose = tmp_path / "infra" / "docker"
    compose.mkdir(parents=True)
    (compose / "docker-compose.yml").write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="vazio"):
        run_yaml("lint", tmp_path)


def test_yaml_lint_ok(tmp_path: Path):
    compose = tmp_path / "infra" / "docker"
    compose.mkdir(parents=True)
    (compose / "docker-compose.yml").write_text("services:\n  redis: {}\n", encoding="utf-8")
    run_yaml("lint", tmp_path)
    run_yaml("security", tmp_path)
    run_yaml("validate", tmp_path)


def test_docker_skip_without_artifacts(tmp_path: Path):
    run_docker("lint", tmp_path)
    run_docker("clean", tmp_path)
    run_area_stage("docker", "lint", tmp_path)


def test_docker_build_skips_locally(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    docker_root = tmp_path / "infra" / "docker"
    docker_root.mkdir(parents=True)
    (docker_root / "Dockerfile").write_text("FROM python:3.13-slim\n", encoding="utf-8")
    (docker_root / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    run_docker("build", tmp_path)
    run_docker("lint", tmp_path)
    run_docker("validate", tmp_path)
    run_docker("security", tmp_path)
    run_docker("test", tmp_path)


def test_dispatch_unknown_stage(tmp_path: Path):
    with pytest.raises(ValueError, match="estagio desconhecido"):
        run_area_stage("docker", "explode", tmp_path)
    with pytest.raises(ValueError, match="desconhecida"):
        run_area_stage("json", "lint", tmp_path)


def test_dispatch_shell_area(tmp_path: Path):
    run_area_stage("shell", "lint", tmp_path)


def test_json_skip_without_config(tmp_path: Path):
    run_json("lint", tmp_path)
    run_yaml("lint", tmp_path)


def test_shell_skip_without_scripts(tmp_path: Path):
    run_shell("lint", tmp_path)
    run_shell("security", tmp_path)
    run_shell("validate", tmp_path)


def test_json_invalid_file(tmp_path: Path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "settings.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError):
        run_json("lint", tmp_path)


def test_yaml_github_workflow(tmp_path: Path):
    workflow = tmp_path / ".github" / "workflows"
    workflow.mkdir(parents=True)
    (workflow / "ci.yml").write_text("name: ci\non: push\njobs: {}\n", encoding="utf-8")
    run_yaml("lint", tmp_path)
    run_yaml("validate", tmp_path)


def test_require_tool_ci_exits(monkeypatch):
    monkeypatch.setenv("CI", "true")
    monkeypatch.setattr("scripts.operations.qa.common.which", lambda name: None)
    with pytest.raises(SystemExit):
        require_tool("hadolint", area="docker")


def test_require_tool_local_skips(monkeypatch):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr("scripts.operations.qa.common.which", lambda name: None)
    assert require_tool("hadolint", area="docker") is None


def test_run_cmd_failure(tmp_path: Path):
    with pytest.raises(subprocess.CalledProcessError):
        run_cmd([sys.executable, "-c", "raise SystemExit(1)"], cwd=tmp_path, description="fail")
    run_cmd([sys.executable, "-c", "raise SystemExit(0)"], cwd=tmp_path, description="ok")


def test_main_config_text(monkeypatch):
    seen: dict[str, object] = {}

    def _fake(stage: str, root: Path, *, kind: str = "all") -> None:
        seen["stage"] = stage
        seen["root"] = root
        seen["kind"] = kind

    monkeypatch.setattr("scripts.operations.clean_workspace.run_config_text", _fake)
    monkeypatch.setattr(sys, "argv", ["clean_workspace.py", "--area", "python", "--stage", "lint", "--config-text"])
    cw.main()
    assert seen["stage"] == "lint"
    assert seen["kind"] == "all"
    monkeypatch.setattr(
        sys,
        "argv",
        ["clean_workspace.py", "--area", "python", "--stage", "lint", "--config-text", "json"],
    )
    cw.main()
    assert seen["kind"] == "json"


def test_main_config_text_requires_python(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["clean_workspace.py", "--area", "docker", "--stage", "lint", "--config-text"],
    )
    with pytest.raises(SystemExit):
        cw.main()


def test_shell_with_script(tmp_path: Path, monkeypatch):
    folder = tmp_path / "linters" / "git-hooks"
    folder.mkdir(parents=True)
    script = folder / "ok.sh"
    script.write_text("#!/bin/sh\necho ok\n", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "scripts.operations.qa.shell.which", lambda name: "/usr/bin/shellcheck" if name == "shellcheck" else None
    )
    monkeypatch.setattr(
        "scripts.operations.qa.shell.run_cmd",
        lambda command, cwd, description: calls.append(list(command)),
    )
    run_shell("lint", tmp_path)
    assert calls[0][0] == "/usr/bin/shellcheck"
    assert "-e" in calls[0] and "SC1091" in calls[0]
    calls.clear()
    monkeypatch.setattr("scripts.operations.qa.shell.which", lambda name: None)

    def _bash(name: str, *, area: str) -> str:
        assert name == "bash"
        assert area == "shell"
        return "/bin/bash"

    monkeypatch.setattr("scripts.operations.qa.shell.require_tool", _bash)
    run_shell("lint", tmp_path)
    run_shell("validate", tmp_path)
    run_shell("build", tmp_path)
    assert any(item[0] == "/bin/bash" for item in calls)


def test_docker_test_requires_compose(tmp_path: Path):
    docker_root = tmp_path / "infra" / "docker"
    docker_root.mkdir(parents=True)
    (docker_root / "Dockerfile").write_text("FROM python:3.13-slim\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        run_docker("test", tmp_path)
