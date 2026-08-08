from pathlib import Path

import torch

from src.application.services.deep_learning.dl_model_checkpoint import should_replace_checkpoint


def test_should_replace_when_deploy_ok_true(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": True, "val_brier": 0.22}, path)
    assert should_replace_checkpoint(path, deploy_ok=True) is True


def test_should_preserve_existing_deploy_ok(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": True, "val_brier": 0.22}, path)
    assert should_replace_checkpoint(path, deploy_ok=False) is False


def test_should_replace_when_existing_not_deploy_ok(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": False, "val_brier": 0.30}, path)
    assert should_replace_checkpoint(path, deploy_ok=False) is True


def test_should_replace_when_missing(tmp_path: Path):
    path = tmp_path / "missing.pth"
    assert should_replace_checkpoint(path, deploy_ok=False) is True


def test_should_replace_when_load_fails(tmp_path: Path, monkeypatch):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"not-a-checkpoint")
    assert should_replace_checkpoint(path, deploy_ok=False) is True


def test_should_replace_when_payload_not_dict(tmp_path: Path, monkeypatch):
    path = tmp_path / "R_10.pth"
    torch.save([1, 2, 3], path)
    assert should_replace_checkpoint(path, deploy_ok=False) is True
