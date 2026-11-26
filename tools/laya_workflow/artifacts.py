"""Integridade dos artefatos reproduziveis do treinamento Laya."""

import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    """Le um artefato JSON UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    """Grava metadados deterministicos sem aceitar NaN."""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def file_digest(path: Path) -> str:
    """Calcula identidade de um arquivo em blocos."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(directory: Path) -> dict:
    """Recusa datasets modificados apos a fixacao do manifesto."""
    manifest = read_json(directory / "manifest.json")
    for name, expected in manifest["files"].items():
        if file_digest(directory / name) != expected:
            raise ValueError(f"Dataset alterado: {name}")
    return manifest


def read_rows(path: Path) -> list[dict]:
    """Le exemplos JSONL preservando a ordem temporal."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
