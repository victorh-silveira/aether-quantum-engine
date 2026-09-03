"""Gates Docker (Hadolint, compose, Trivy, build)."""

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
            run_cmd([hadolint, str(path)], cwd=root, description=f"Hadolint {path.relative_to(root)}")
        return
    if stage == "validate":
        docker = require_tool("docker", area="docker")
        if docker is None:
            return
        if not compose.is_file():
            skip("docker", "compose ausente")
            return
        run_cmd(
            [docker, "compose", "-f", str(compose), "config", "-q"],
            cwd=root,
            description="Docker Compose config",
        )
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
        for path in files:
            tag = f"aether-ci-{path.parent.name}:local".lower()
            run_cmd(
                [docker, "build", "-f", str(path), "-t", tag, str(path.parent)],
                cwd=root,
                description=f"Docker build {path.relative_to(root)}",
            )
        return
    skip("docker", f"estagio {stage} nao aplicavel")
