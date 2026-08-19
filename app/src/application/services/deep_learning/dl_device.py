"""Selecao de dispositivo PyTorch (CPU/CUDA) para treino e inferencia."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn


logger = logging.getLogger("AETH")

_DEVICE_LOGGED: set[str] = set()


def _config_device(dl_config: dict[str, Any] | None, key: str, default: str = "auto") -> str:
    """Le chave de dispositivo da config DL com fallback para o valor padrao."""
    if not isinstance(dl_config, dict):
        return default
    raw = dl_config.get(key)
    if raw is None:
        return default
    return str(raw).strip().lower()


def resolve_torch_device(dl_config: dict[str, Any] | None, *, kind: str) -> torch.device:
    """Resolve cpu ou cuda conforme config e disponibilidade de GPU."""
    config_key = "training_device" if kind == "training" else "inference_device"
    preference = _config_device(dl_config, config_key, "auto")
    if preference in ("cpu", "off", "false", "0"):
        return torch.device("cpu")
    if preference.startswith("cuda"):
        if torch.cuda.is_available():
            if ":" in preference:
                return torch.device(preference)
            return torch.device("cuda:0")
        logger.warning("DL: %s pediu CUDA mas GPU indisponivel; usando CPU.", kind)
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda:0")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    """Rotulo legivel do dispositivo para logs."""
    if device.type != "cuda":
        return "cpu"
    index = device.index if device.index is not None else 0
    try:
        name = torch.cuda.get_device_name(index)
    except Exception:
        name = "cuda"
    return f"cuda:{index} ({name})"


def log_device_once(device: torch.device, *, context: str) -> None:
    """Registra dispositivo DL uma vez por contexto (treino/inferencia)."""
    tag = f"{context}:{device.type}:{getattr(device, 'index', 0)}"
    if tag in _DEVICE_LOGGED:
        return
    _DEVICE_LOGGED.add(tag)
    logger.info("DL: %s em %s", context, device_label(device))


def place_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Move modulo para o dispositivo informado."""
    return model.to(device)


def tensor_from_numpy(array, device: torch.device, *, dtype=torch.float32) -> torch.Tensor:
    """Converte ndarray para tensor no dispositivo alvo."""
    arr = np.asarray(array, dtype=np.float32)
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.as_tensor(arr, dtype=dtype, device=device)
