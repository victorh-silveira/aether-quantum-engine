"""Ponte de integracao nativa entre o runtime Python e o crate Rust aether_core_rs."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

try:
    aether_core_rs = importlib.import_module("aether_core_rs")
    _RUST_AVAILABLE = True
except ImportError:
    aether_core_rs = None
    _RUST_AVAILABLE = False


class DisruptorRingBuffer:
    """Fallback Python/NumPy quando a extensao Rust aether_core_rs nao esta compilada."""

    def __init__(self, capacity: int = 4096) -> None:
        """Inicializa arrays contiguos e mascara de bits para capacidade potencia de 2."""
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            msg = f"Capacidade deve ser potencia de 2 positiva, recebido {capacity}"
            raise ValueError(msg)
        self._capacity: int = capacity
        self._mask: int = capacity - 1
        self._sequence: int = 0
        self._epochs: np.ndarray = np.zeros(capacity, dtype=np.int64)
        self._prices: np.ndarray = np.zeros(capacity, dtype=np.float64)

    @property
    def capacity(self) -> int:
        """Retorna a capacidade total de slots do buffer."""
        return self._capacity

    @property
    def count(self) -> int:
        """Retorna a quantidade efetiva de elementos contidos no buffer."""
        return min(self._sequence, self._capacity)

    @property
    def sequence(self) -> int:
        """Retorna o cursor absoluto da sequencia monotonicamente crescente."""
        return self._sequence

    def push_tick(self, epoch_ms: int, price: float) -> int:
        """Insere um novo tick no slot calculado por mascara de bits em O(1)."""
        slot = self._sequence & self._mask
        self._epochs[slot] = epoch_ms
        self._prices[slot] = price
        current_seq = self._sequence
        self._sequence += 1
        return current_seq

    def latest_tick(self) -> tuple[int, float] | None:
        """Retorna o par epoch_ms e preco do tick mais recente ou None se vazio."""
        if self._sequence == 0:
            return None
        latest_slot = (self._sequence - 1) & self._mask
        return int(self._epochs[latest_slot]), float(self._prices[latest_slot])

    def latest_window(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Extrai os ultimos n ticks em ordem cronologica estrita."""
        if n <= 0 or self._sequence == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        target_n = min(n, self.count)
        start_seq = self._sequence - target_n
        indices = np.bitwise_and(np.arange(start_seq, self._sequence, dtype=np.int64), self._mask)
        return self._epochs[indices].copy(), self._prices[indices].copy()

    def get_prices_array(self, n: int) -> np.ndarray:
        """Extrai exclusivamente a serie temporal dos ultimos n precos em ordem temporal."""
        if n <= 0 or self._sequence == 0:
            return np.empty(0, dtype=np.float64)
        target_n = min(n, self.count)
        start_seq = self._sequence - target_n
        indices = np.bitwise_and(np.arange(start_seq, self._sequence, dtype=np.int64), self._mask)
        return self._prices[indices].copy()

    def clear(self) -> None:
        """Reseta os cursores do buffer sem desalocar os arrays subjacentes."""
        self._sequence = 0
        self._epochs.fill(0)
        self._prices.fill(0.0)


def is_rust_core_available() -> bool:
    """Informa se a extensao nativa em Rust esta compilada e acessivel."""
    return _RUST_AVAILABLE


def create_native_ring_buffer(capacity: int = 4096) -> Any:
    """Instancia ring buffer compilado em Rust ou recorre ao fallback contiguo."""
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
