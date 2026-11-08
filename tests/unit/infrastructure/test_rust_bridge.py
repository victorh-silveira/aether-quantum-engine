"""Testes unitarios para a ponte de extensao nativa Rust."""

from unittest.mock import MagicMock

import src.infrastructure.native.rust_bridge as rb
from src.infrastructure.native.rust_bridge import (
    calculate_fast_realized_volatility,
    create_native_ring_buffer,
    is_rust_core_available,
)


def test_rust_bridge_is_available_returns_bool():
    available = is_rust_core_available()
    assert isinstance(available, bool)


def test_rust_bridge_create_ring_buffer_push_and_retrieval():
    buf = create_native_ring_buffer(capacity=16)
    assert buf.count == 0

    buf.push_tick(1000, 50.0)
    buf.push_tick(1001, 51.0)
    assert buf.count == 2
    assert buf.latest_tick() == (1001, 51.0)


def test_rust_bridge_calculate_realized_volatility():
    prices = [100.0, 102.0, 101.0, 103.0]
    rv = calculate_fast_realized_volatility(prices)
    assert isinstance(rv, float)
    assert rv > 0.0

    assert calculate_fast_realized_volatility([100.0]) == 0.0
    assert calculate_fast_realized_volatility([]) == 0.0


def test_rust_bridge_mock_native_available(monkeypatch):
    class MockRustRing:
        def __init__(self, capacity: int):
            self.capacity = capacity

    mock_crate = MagicMock()
    mock_crate.RustRingBuffer = MockRustRing
    mock_crate.compute_realized_volatility_rust.return_value = 0.42

    monkeypatch.setattr(rb, "_RUST_AVAILABLE", True)
    monkeypatch.setattr(rb, "aether_core_rs", mock_crate)

    buf = create_native_ring_buffer(64)
    assert isinstance(buf, MockRustRing)
    assert buf.capacity == 64

    val = calculate_fast_realized_volatility([10.0, 11.0, 12.0])
    assert val == 0.42


def test_rust_bridge_mock_native_exceptions(monkeypatch):
    mock_crate = MagicMock()
    mock_crate.RustRingBuffer.side_effect = RuntimeError("Erro de alocacao C")
    mock_crate.compute_realized_volatility_rust.side_effect = RuntimeError("Erro numerico Rust")

    monkeypatch.setattr(rb, "_RUST_AVAILABLE", True)
    monkeypatch.setattr(rb, "aether_core_rs", mock_crate)

    buf = create_native_ring_buffer(32)
    assert buf is not None
    assert buf.capacity == 32

    val = calculate_fast_realized_volatility([100.0, 105.0])
    assert isinstance(val, float)
    assert val > 0.0


def test_rust_bridge_reload_with_mock(monkeypatch):
    import importlib
    import sys
    from types import ModuleType

    mock_crate = ModuleType("aether_core_rs")
    monkeypatch.setitem(sys.modules, "aether_core_rs", mock_crate)
    importlib.reload(rb)
    assert rb.is_rust_core_available() is True

    monkeypatch.delitem(sys.modules, "aether_core_rs", raising=False)
    importlib.reload(rb)
