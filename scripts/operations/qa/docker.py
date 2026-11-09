"""Gates Docker (Hadolint, compose, Trivy, build)."""

import json
import os
from pathlib import Path

from scripts.operations.qa.common import is_ci, require_tool, run_cmd, skip


def _dockerfiles(root: Path) -> list[Path]:
    docker_root = root / "infra" / "docker"
    if not docker_root.is_dir():
        return []
    found: list[Path] = []
    for path in docker_root.rglob("Dockerfile"):
        found.append(path)
    for path in docker_root.rglob("Dockerfile.*"):
        found.append(path)
    return sorted(found)


def _compose_file(root: Path) -> Path:
    return root / "infra" / "docker" / "docker-compose.yml"


def _validate_grafana_dashboards(root: Path) -> None:
    dashboards_dir = root / "infra" / "docker" / "grafana" / "indicator-dashboards"
    if not dashboards_dir.is_dir():
        return
    files = sorted(dashboards_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Nenhum dashboard Grafana encontrado em {dashboards_dir}")
    for file in files:
        try:
            json.loads(file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Dashboard Grafana corrompido em {file}: {exc}") from exc
    print(f"[docker] grafana dashboards ok ({len(files)} arquivos)")


def run_docker(stage: str, root: Path) -> None:
    """Executa um estagio da area docker."""
    files = _dockerfiles(root)
    compose = _compose_file(root)
    if not files and not compose.is_file():
        skip("docker", "nenhum artefato")
        return
    if stage == "lint":
        hadolint = require_tool("hadolint", area="docker")
        if hadolint is None:
            return
        for path in files:
            run_cmd(
                [hadolint, "--ignore", "DL3008", "--ignore", "DL3022", str(path)],
                cwd=root,
                description=f"Hadolint {path.relative_to(root)}",
            )
        return
    if stage == "validate":
        docker = require_tool("docker", area="docker")
        if docker is None:
            return
        if not compose.is_file():
            skip("docker", "compose ausente")
            return
        compose_env = dict(os.environ)
        compose_env["COMPOSE_PROFILES"] = "core,telemetry,indicator"
        run_cmd(
            [docker, "compose", "-f", str(compose), "--project-directory", str(compose.parent), "config", "-q"],
            cwd=root,
            description="Docker Compose config (profiles core,telemetry,indicator)",
            env=compose_env,
        )
        _validate_grafana_dashboards(root)
        return
    if stage == "security":
        trivy = require_tool("trivy", area="docker")
        if trivy is None:
            return
        run_cmd(
            [
                trivy,
                "fs",
                "--exit-code",
                "1",
                "--severity",
                "HIGH,CRITICAL",
                "--ignore-unfixed",
                str(root / "infra" / "docker"),
            ],
            cwd=root,
            description="Trivy fs infra/docker",
        )
        return
    if stage == "test":
        if not compose.is_file():
            raise FileNotFoundError("infra/docker/docker-compose.yml ausente")
        missing = [path for path in files if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Dockerfiles ausentes: {missing}")
        print("[docker] smoke: compose e Dockerfiles presentes")
        return
    if stage == "build":
        if not is_ci():
            skip("docker", "build reservado ao CI")
            return
        docker = require_tool("docker", area="docker")
        if docker is None:
            return
        docker_root = root / "infra" / "docker"
        ml_common = docker_root / "ml_common"
        for path in files:
            tag = f"aether-ci-{path.parent.name}:local".lower()
            cmd = [docker, "build", "-f", str(path), "-t", tag]
            if path.parent.name in {"meta-classifier", "loss-classifier"} and ml_common.is_dir():
                cmd.extend(["--build-context", f"ml_common={ml_common}"])
            cmd.append(str(path.parent))
            run_cmd(
                cmd,
                cwd=root,
                description=f"Docker build {path.relative_to(root)}",
            )
        return
    skip("docker", f"estagio {stage} nao aplicavel")
