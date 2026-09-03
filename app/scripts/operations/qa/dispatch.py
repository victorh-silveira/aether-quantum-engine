"""Despacho de area/estagio fora do Python."""

from pathlib import Path

from scripts.operations.qa.common import STAGES
from scripts.operations.qa.docker import run_docker
from scripts.operations.qa.json_area import run_json
from scripts.operations.qa.shell import run_shell
from scripts.operations.qa.yaml_area import run_yaml


def run_config_text(stage: str, root: Path, *, kind: str = "all") -> None:
    """Valida JSON de config e/ou YAML operacional dentro do job Python."""
    if kind not in {"all", "json", "yaml"}:
        raise ValueError(f"config-text desconhecido: {kind}")
    if kind in {"all", "json"}:
        run_json(stage, root)
    if kind in {"all", "yaml"}:
        run_yaml(stage, root)


def run_area_stage(area: str, stage: str, root: Path) -> None:
    """Roteia area nao-python para o modulo correspondente."""
    if stage not in STAGES:
        raise ValueError(f"estagio desconhecido: {stage}")
    if area == "docker":
        run_docker(stage, root)
        return
    if area == "shell":
        run_shell(stage, root)
        return
    raise ValueError(f"area desconhecida: {area}")
