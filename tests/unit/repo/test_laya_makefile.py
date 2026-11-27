"""Contratos Make do fluxo Laya sem iniciar treino ou enviar ordens."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "target,expected",
    [
        ("app-train", "laya_workflow train"),
        ("app-evaluate", "laya_workflow evaluate"),
        ("app-promote", "laya_workflow promote"),
        ("app-prepare", "laya_workflow prepare"),
        ("app-train-dry-run", "--dry-run"),
        ("app-run", "run.py"),
    ],
)
def test_make_routes_commands_and_preserves_paths_with_spaces(target, expected):
    root = Path(__file__).resolve().parents[3]
    args = [
        "make",
        "--no-print-directory",
        "--dry-run",
        target,
        "LAYA_DATA=.cache/dados com espacos",
        "LAYA_RUN=.cache/nova run",
        "LAYA_TRAIN_PROFILE=head",
    ]
    if os.name == "nt":
        wsl_root = "/mnt/" + root.drive[0].lower() + root.as_posix()[2:]
        command = ["wsl.exe", "bash", "-lc", f"cd {shlex.quote(wsl_root)} && {shlex.join(args)}"]
    else:
        command = args
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=True, timeout=30)
    assert expected in result.stdout
    if target in {"app-train", "app-evaluate", "app-promote", "app-prepare", "app-train-dry-run"}:
        assert '--data ".cache/dados com espacos"' in result.stdout
    if target in {"app-train", "app-evaluate", "app-promote"}:
        assert '--run ".cache/nova run"' in result.stdout
    if target == "app-train":
        assert '--profile "head"' in result.stdout
        assert '--preset "fast"' in result.stdout
        assert '--warm-start-auto ".cache/laya-r10-gpu-run/seed-17/checkpoint_latest"' in result.stdout
    if target == "app-train-dry-run":
        assert '--preset "fast"' in result.stdout
    if target in {"app-run", "app-train"}:
        assert result.stdout.index("laya_cuda_check --mode runtime") < result.stdout.index(expected)
