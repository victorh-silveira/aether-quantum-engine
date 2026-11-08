"""Ponte de integracao nativa entre o runtime Python e o crate Rust aether_core_rs."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np

from src.infrastructure.handlers.disruptor_ring_buffer import DisruptorRingBuffer


logger = logging.getLogger(__name__)

try:
    aether_core_rs = importlib.import_module("aether_core_rs")
    _RUST_AVAILABLE = True
except ImportError:
    aether_core_rs = None
    _RUST_AVAILABLE = False


def is_rust_core_available() -> bool:
    """Informa se a extensao nativa em Rust esta compilada e acessivel."""
    return _RUST_AVAILABLE


def create_native_ring_buffer(capacity: int = 4096) -> Any:
    """Instancia ring buffer compilado em Rust ou recorre a versao NumPy contigua."""
    if _RUST_AVAILABLE and aether_core_rs is not None:
        try:
            return aether_core_rs.RustRingBuffer(int(capacity))
        except Exception as exc:
            logger.debug("Falha ao instanciar RustRingBuffer: %s", exc)
            return DisruptorRingBuffer(capacity=capacity)
    return DisruptorRingBuffer(capacity=capacity)


def calculate_fast_realized_volatility(prices: list[float] | np.ndarray) -> float:
    """Calcula volatilidade realizada via funcao nativa Rust ou vetorizacao NumPy."""
    if _RUST_AVAILABLE and aether_core_rs is not None:
        try:
            p_list = [float(p) for p in prices]
            return float(aether_core_rs.compute_realized_volatility_rust(p_list))
        except Exception as exc:
            logger.debug("Falha na computacao Rust da volatilidade realizada: %s", exc)

    arr = np.asarray(prices, dtype=np.float64)
    valid = arr[arr > 0]
    if len(valid) < 2:
        return 0.0
    rets = np.diff(np.log(valid))
    return float(np.sqrt(np.sum(rets**2)))
