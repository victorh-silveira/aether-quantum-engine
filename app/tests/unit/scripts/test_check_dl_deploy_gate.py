"""Gate de deploy/ACC apos treino DL."""

from pathlib import Path
from unittest.mock import patch

from scripts.operations.check_dl_deploy_gate import evaluate_checkpoint, main


def test_evaluate_checkpoint_rejects_low_acc(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch("torch.load", return_value={"val_accuracy": 0.52, "deploy_ok": True}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is False
    assert "val_acc" in msg


def test_evaluate_checkpoint_rejects_deploy_false(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch("torch.load", return_value={"val_accuracy": 0.60, "deploy_ok": False}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is False
    assert "deploy_ok=false" in msg


def test_evaluate_checkpoint_accepts_senior(tmp_path: Path):
    path = tmp_path / "R_10.pth"
    path.write_bytes(b"x")
    with patch("torch.load", return_value={"val_accuracy": 0.55, "deploy_ok": True}):
        ok, msg = evaluate_checkpoint(path, soft_min=0.53)
    assert ok is True
    assert "deploy_ok=true" in msg


def test_main_fails_when_checkpoint_missing(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("sys.argv", ["check_dl_deploy_gate.py", "--symbols", "R_10"])
    with (
        patch(
            "scripts.operations.check_dl_deploy_gate._checkpoint_paths",
            return_value=[tmp_path / "missing.pth"],
        ),
        patch("scripts.operations.check_dl_deploy_gate._load_settings", return_value={}),
        patch("scripts.operations.check_dl_deploy_gate._soft_min_acc", return_value=0.53),
    ):
        assert main() == 1
