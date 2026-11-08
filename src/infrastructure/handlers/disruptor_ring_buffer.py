"""Ring buffer circular de alta performance baseado no padrao LMAX Disruptor."""

from __future__ import annotations

import numpy as np


class DisruptorRingBuffer:
    """Buffer circular contiguo com indexacao bitwise para ingestao de micro-ticks."""

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
        available = self.count
        target_n = min(n, available)
        start_seq = self._sequence - target_n
        indices = np.bitwise_and(np.arange(start_seq, self._sequence, dtype=np.int64), self._mask)
        return self._epochs[indices].copy(), self._prices[indices].copy()

    def get_prices_array(self, n: int) -> np.ndarray:
        """Extrai exclusivamente a serie temporal dos ultimos n precos em ordem temporal."""
        if n <= 0 or self._sequence == 0:
            return np.empty(0, dtype=np.float64)
        available = self.count
        target_n = min(n, available)
        start_seq = self._sequence - target_n
        indices = np.bitwise_and(np.arange(start_seq, self._sequence, dtype=np.int64), self._mask)
        return self._prices[indices].copy()

    def clear(self) -> None:
        """Reseta os cursores do buffer sem desalocar os arrays subjacentes."""
        self._sequence = 0
        self._epochs.fill(0)
        self._prices.fill(0.0)
