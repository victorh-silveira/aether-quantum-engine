"""Carregamento tardio e liberacao de heap para testes que usam PyTorch."""

from __future__ import annotations

import gc
import importlib
from typing import Any


def load_torch() -> tuple[Any, Any]:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    return torch, nn


def release_torch_heap() -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
