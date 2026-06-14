from unittest.mock import patch

import numpy as np
import torch

from src.application.services.deep_learning import dl_device
from src.application.services.deep_learning.dl_device import (
    device_label,
    log_device_once,
    place_model,
    resolve_torch_device,
    tensor_from_numpy,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.model import create_direction_model


def test_resolve_torch_device_cpu_forced():
    device = resolve_torch_device({"training_device": "cpu"}, kind="training")
    assert device.type == "cpu"


def test_resolve_torch_device_none_config_defaults_cpu():
    with patch("torch.cuda.is_available", return_value=False):
        device = resolve_torch_device(None, kind="training")
    assert device.type == "cpu"


def test_resolve_torch_device_missing_key_defaults_auto():
    with patch("torch.cuda.is_available", return_value=False):
        device = resolve_torch_device({}, kind="training")
    assert device.type == "cpu"


def test_resolve_torch_device_auto_without_cuda():
    with patch("torch.cuda.is_available", return_value=False):
        device = resolve_torch_device({"training_device": "auto"}, kind="training")
    assert device.type == "cpu"


def test_resolve_torch_device_auto_with_cuda():
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.get_device_name", return_value="RTX 4060"),
    ):
        device = resolve_torch_device({"inference_device": "auto"}, kind="inference")
    assert device.type == "cuda"
    assert device.index == 0


def test_resolve_torch_device_cuda_without_index():
    with patch("torch.cuda.is_available", return_value=True):
        device = resolve_torch_device({"training_device": "cuda"}, kind="training")
    assert str(device) == "cuda:0"


def test_resolve_torch_device_cuda_unavailable_falls_back():
    with patch("torch.cuda.is_available", return_value=False):
        device = resolve_torch_device({"training_device": "cuda"}, kind="training")
    assert device.type == "cpu"


def test_device_label_cpu():
    assert device_label(torch.device("cpu")) == "cpu"


def test_device_label_cuda():
    with patch("torch.cuda.get_device_name", return_value="RTX 4060"):
        label = device_label(torch.device("cuda:0"))
    assert "RTX 4060" in label


def test_device_label_cuda_when_name_lookup_fails():
    with patch("torch.cuda.get_device_name", side_effect=RuntimeError("no gpu")):
        label = device_label(torch.device("cuda:0"))
    assert label == "cuda:0 (cuda)"


def test_log_device_once_deduplicates():
    dl_device._DEVICE_LOGGED.clear()
    log_device_once(torch.device("cpu"), context="treino")
    log_device_once(torch.device("cpu"), context="treino")
    assert len(dl_device._DEVICE_LOGGED) == 1


def test_resolve_explicit_cuda_index():
    with patch("torch.cuda.is_available", return_value=True):
        device = resolve_torch_device({"training_device": "cuda:1"}, kind="training")
    assert str(device) == "cuda:1"


def test_place_model_and_tensor_from_numpy():
    model = create_direction_model(arch="tcn")
    place_model(model, torch.device("cpu"))
    batch = np.zeros((2, 8, FEATURE_DIM), dtype=np.float32)
    tensor = tensor_from_numpy(batch, torch.device("cpu"))
    assert tensor.device.type == "cpu"
    assert tensor.shape[0] == 2


def test_tensor_from_numpy_sanitizes_non_finite():
    batch = np.array([[[np.nan, np.inf, -np.inf] + [0.0] * (FEATURE_DIM - 3)]], dtype=np.float32)
    tensor = tensor_from_numpy(batch, torch.device("cpu"))
    assert torch.isfinite(tensor).all()
