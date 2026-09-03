"""Constantes e helpers dos gates de QA."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


AREAS = ("python", "docker", "shell")
STAGES = ("lint", "validate", "security", "test", "build", "clean")


def is_ci() -> bool:
    """True quando a variavel de ambiente CI indica GitHub Actions."""
    return str(os.environ.get("CI", "")).strip().lower() in {"1", "true", "yes"}


def skip(area: str, reason: str) -> None:
    """Registra skip da area sem falhar."""
    print(f"[{area}] skip: {reason}")


def posix_for_tool(path: Path) -> str:
    """Converte path Windows para o equivalente /mnt no bash da WSL."""
    posix = path.resolve().as_posix()
    if len(posix) >= 2 and posix[1] == ":":
        drive = posix[0].lower()
        return f"/mnt/{drive}{posix[2:]}"
    return posix


def which(name: str) -> str | None:
    """Resolve executavel no PATH."""
    found = shutil.which(name)
    return found if found else None


def require_tool(name: str, *, area: str) -> str | None:
    """Exige ferramenta no CI; localmente faz skip se ausente."""
    path = which(name)
    if path:
        return path
    if is_ci():
        print(f"[ERRO] {name} obrigatorio no CI para area {area}")
        sys.exit(1)
    skip(area, f"{name} ausente")
    return None


def run_cmd(command: list[str], *, cwd: Path, description: str) -> None:
    """Executa comando e propaga codigo de saida diferente de zero."""
    print(f"\n>>> Executando: {description}")
    completed = subprocess.run(command, check=False, text=True, cwd=str(cwd), shell=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
