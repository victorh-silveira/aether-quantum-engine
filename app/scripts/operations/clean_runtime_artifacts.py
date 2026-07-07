"""Remove artefatos de runtime regeneraveis por treino e execucao."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


RemoveFn = Callable[[Path], None]
PRESERVED_DATA_CHILDREN = frozenset({"deriv"})
DOCKER_BIND_MOUNTS_RELATIVE = (
    Path("infra") / "docker" / "triton-models",
    Path("infra") / "docker" / "meta-models",
)


def docker_bind_mount_roots(repo_root: Path) -> tuple[Path, ...]:
    """Retorna bind mounts do docker-compose que nao devem ser apagados pelo make clean."""
    return tuple(repo_root / relative for relative in DOCKER_BIND_MOUNTS_RELATIVE)


def triton_models_root(repo_root: Path) -> Path:
    """Retorna o bind mount do repositorio Triton preservado pelo make clean."""
    return repo_root / DOCKER_BIND_MOUNTS_RELATIVE[0]


def meta_models_root(repo_root: Path) -> Path:
    """Retorna o bind mount do meta-classificador preservado pelo make clean."""
    return repo_root / DOCKER_BIND_MOUNTS_RELATIVE[1]


def is_docker_bind_mount(path: Path, repo_root: Path) -> bool:
    """True quando o caminho esta dentro de um bind mount montado nos containers."""
    try:
        resolved = path.resolve()
        repo = repo_root.resolve()
        relative = resolved.relative_to(repo)
    except ValueError:
        return False
    relative_posix = relative.as_posix()
    for mount in DOCKER_BIND_MOUNTS_RELATIVE:
        mount_posix = mount.as_posix()
        if relative_posix == mount_posix or relative_posix.startswith(f"{mount_posix}/"):
            return True
    return False


def clean_repo_data(data_root: Path, safe_remove: RemoveFn) -> None:
    """Remove checkpoints DL, TorchScript local e estado de sessao; preserva data/deriv."""
    if not data_root.exists():
        return
    for child in list(data_root.iterdir()):
        if child.name in PRESERVED_DATA_CHILDREN:
            continue
        safe_remove(child)


def clean_runtime_artifacts(repo_root: Path, safe_remove: RemoveFn) -> None:
    """Limpa dados locais de run/treino sem tocar bind mounts Docker."""
    clean_repo_data(repo_root / "data", safe_remove)
