"""Integridade de pesos aprovados e rejeicao de alteracoes posteriores."""

import json

import pytest

from src.infrastructure.inference.laya_checkpoint import checkpoint_digest, resolve_approved_checkpoint


def test_checkpoint_approval_integrity(tmp_path):
    assert resolve_approved_checkpoint("base", "") == ("base", "base")
    with pytest.raises(ValueError, match="sem pesos"):
        checkpoint_digest(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"test weights")
    digest = checkpoint_digest(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps({"approved": True, "checkpoint_sha256": digest}))
    assert resolve_approved_checkpoint("base", str(tmp_path)) == (str(tmp_path), digest)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="alterado"):
        resolve_approved_checkpoint("base", str(tmp_path))
