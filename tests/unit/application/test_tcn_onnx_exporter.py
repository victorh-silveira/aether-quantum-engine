"""Testes unitarios para o exportador de modelos TCN para ONNX."""

from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from src.application.services.deep_learning.tcn_onnx_exporter import (
    export_checkpoint_to_onnx,
    export_tcn_to_onnx,
)


class DummyTcn(nn.Module):
    def __init__(self, feature_dim: int = 14):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, x):
        return self.linear(x[:, -1, :])


def test_export_tcn_to_onnx_success(tmp_path: Path, monkeypatch):
    model = DummyTcn(feature_dim=14)
    target = tmp_path / "test_model.onnx"

    def fake_export(_m, _i, path, **_kwargs):
        Path(path).write_bytes(b"ONNX_MOCK_PAYLOAD")

    monkeypatch.setattr(torch.onnx, "export", fake_export)
    success = export_tcn_to_onnx(model, target, lookback=16, feature_dim=14)
    assert success
    assert target.exists()
    assert target.stat().st_size > 0


def test_export_tcn_to_onnx_failure(tmp_path: Path, monkeypatch):
    model = DummyTcn(feature_dim=14)
    target = tmp_path / "fail_model.onnx"

    def broken_export(*_args, **_kwargs):
        raise RuntimeError("Erro simulado do exportador ONNX")

    monkeypatch.setattr(torch.onnx, "export", broken_export)
    assert not export_tcn_to_onnx(model, target)


def test_export_checkpoint_to_onnx_missing_file():
    result = export_checkpoint_to_onnx(Path("non/existent/checkpoint.pt"))
    assert result is None


def test_export_checkpoint_to_onnx_load_none(tmp_path: Path):
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"dummy")

    with patch("src.application.services.deep_learning.tcn_onnx_exporter.load_model_checkpoint", return_value=None):
        assert export_checkpoint_to_onnx(ckpt) is None


def test_export_checkpoint_to_onnx_success_and_failure(tmp_path: Path):
    ckpt = tmp_path / "valid.pt"
    ckpt.write_bytes(b"valid")
    model = DummyTcn(feature_dim=14)
    fake_tuple = (model, None, None, None, 32, None, None, None, None, None)

    with patch(
        "src.application.services.deep_learning.tcn_onnx_exporter.load_model_checkpoint", return_value=fake_tuple
    ):
        with patch("src.application.services.deep_learning.tcn_onnx_exporter.export_tcn_to_onnx", return_value=True):
            out = export_checkpoint_to_onnx(ckpt)
            assert out is not None
            assert out.suffix == ".onnx"

        with patch("src.application.services.deep_learning.tcn_onnx_exporter.export_tcn_to_onnx", return_value=False):
            assert export_checkpoint_to_onnx(ckpt) is None
