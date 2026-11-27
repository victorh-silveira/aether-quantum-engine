"""Identidade verificavel de checkpoints promovidos para o motor Laya."""

import hashlib
import json
from pathlib import Path


def checkpoint_digest(directory: Path) -> str:
    """Identifica pesos, tokenizer, perguntas e calibracao sem incluir aprovacao."""
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name not in {"approval.json", "resume.json"}
        and not {"checkpoint_latest", "resume"}.intersection(path.relative_to(directory).parts)
    )
    if not files or not (directory / "model.safetensors").is_file():
        raise ValueError("Checkpoint sem pesos locais")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def resolve_approved_checkpoint(base: str, approved: str) -> tuple[str, str]:
    """Usa o modelo base por padrao; caminho configurado exige aprovacao integra."""
    if not approved:
        return base, base
    directory = Path(approved)
    manifest = json.loads((directory / "approval.json").read_text(encoding="utf-8"))
    version = checkpoint_digest(directory)
    if (manifest.get("approved") is not True and manifest.get("operator_override") is not True) or manifest.get(
        "checkpoint_sha256"
    ) != version:
        raise ValueError("Checkpoint local sem aprovacao valida ou alterado apos promocao")
    return str(directory), version
