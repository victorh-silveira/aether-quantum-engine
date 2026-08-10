from pathlib import Path

import torch

from src.application.services.deep_learning.dl_model_checkpoint import should_replace_checkpoint


def test_should_replace_when_deploy_ok_true(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": True, "val_brier": 0.22}, path)
    assert should_replace_checkpoint(path, deploy_ok=True) is True


def test_should_replace_even_when_existing_deploy_ok(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": True, "val_brier": 0.22}, path)
    assert should_replace_checkpoint(path, deploy_ok=False) is True


def test_should_replace_when_existing_not_deploy_ok(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": False, "val_brier": 0.30, "val_accuracy": 0.50}, path)
    assert should_replace_checkpoint(path, deploy_ok=False, val_accuracy=0.52) is True


def test_should_replace_even_when_old_acc_better(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": False, "val_accuracy": 0.556}, path)
    assert should_replace_checkpoint(path, deploy_ok=False, val_accuracy=0.5176) is True


def test_should_replace_when_val_accuracy_none(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": False, "val_accuracy": 0.50}, path)
    assert should_replace_checkpoint(path, deploy_ok=False, val_accuracy=None) is True


def test_checkpoint_meta_ready_true_when_deploy_ok(tmp_path: Path):
    from src.application.services.deep_learning.dl_model_checkpoint import checkpoint_meta_ready

    path = tmp_path / "R_10.pth"
    torch.save({"deploy_ok": True, "val_accuracy": 0.556}, path)
    assert checkpoint_meta_ready(path) is True


def test_checkpoint_meta_ready_false_when_missing(tmp_path: Path):
    from src.application.services.deep_learning.dl_model_checkpoint import checkpoint_meta_ready

    assert checkpoint_meta_ready(tmp_path / "missing.pth") is False


def test_checkpoint_meta_ready_false_on_corrupt_or_non_dict(tmp_path: Path):
    from src.application.services.deep_learning.dl_model_checkpoint import checkpoint_meta_ready

    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"not-a-checkpoint")
    assert checkpoint_meta_ready(bad) is False
    not_dict = tmp_path / "list.pth"
    torch.save([1, 2, 3], not_dict)
    assert checkpoint_meta_ready(not_dict) is False


def test_checkpoint_meta_ready_soft_gate_without_flag(tmp_path: Path):
    from src.application.services.deep_learning.dl_model_checkpoint import checkpoint_meta_ready

    path = tmp_path / "R_10.pth"
    torch.save(
        {
            "deploy_ok": False,
            "val_accuracy": 0.556,
            "val_brier": 0.24,
            "label_call_frac": 0.44,
            "pred_call_frac": 0.66,
            "minority_recall": 0.38,
        },
        path,
    )
    assert checkpoint_meta_ready(path) is True


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
