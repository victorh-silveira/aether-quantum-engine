from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from src.infrastructure.storage.torchscript_sanity import verify_torchscript_artifact


class _TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(19, 1)

    def forward(self, x):
        return self.fc(x[:, -1, :])


def test_verify_torchscript_artifact_success(tmp_path):
    path = tmp_path / "m_ts.pt"
    model = _TinyNet()
    model.eval()
    example = torch.zeros(1, 48, 19)
    traced = torch.jit.trace(model, example, strict=False)
    traced.save(str(path))
    verify_torchscript_artifact(path, lookback=48, feature_dim=19)


def test_verify_torchscript_nan_raises(tmp_path):
    path = tmp_path / "bad_ts.pt"

    class _NanNet(nn.Module):
        def forward(self, _x):
            return torch.tensor([[float("nan")]])

    traced = torch.jit.trace(_NanNet(), torch.zeros(1, 48, 19), strict=False)
    traced.save(str(path))
    with pytest.raises(RuntimeError, match="NaN"):
        verify_torchscript_artifact(path, lookback=48, feature_dim=19)


def test_verify_torchscript_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="ausente"):
        verify_torchscript_artifact(tmp_path / "missing.pt", lookback=48, feature_dim=19)


def test_verify_torchscript_tuple_output(tmp_path):
    path = tmp_path / "tuple_ts.pt"

    class _TupleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(19, 1)

        def forward(self, x):
            return (self.fc(x[:, -1, :]),)

    traced = torch.jit.trace(_TupleNet(), torch.zeros(1, 48, 19), strict=False)
    traced.save(str(path))
    verify_torchscript_artifact(path, lookback=48, feature_dim=19)


def test_verify_torchscript_invalid_type_raises(tmp_path):
    path = tmp_path / "bad_type.pt"
    path.write_bytes(b"x")
    model = MagicMock()
    model.eval.return_value = None
    model.return_value = 1
    with patch("torch.jit.load", return_value=model), pytest.raises(RuntimeError, match="tipo invalido"):
        verify_torchscript_artifact(path, lookback=48, feature_dim=19)


def test_verify_torchscript_invalid_shape_raises(tmp_path):
    path = tmp_path / "bad_shape.pt"

    class _EmptyNet(nn.Module):
        def forward(self, _x):
            return torch.zeros(0)

    traced = torch.jit.trace(_EmptyNet(), torch.zeros(1, 48, 19), strict=False)
    traced.save(str(path))
    with pytest.raises(RuntimeError, match="shape invalida"):
        verify_torchscript_artifact(path, lookback=48, feature_dim=19)
