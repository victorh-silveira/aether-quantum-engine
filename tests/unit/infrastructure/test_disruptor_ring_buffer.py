"""Testes unitarios para o modulo disruptor_ring_buffer."""

import numpy as np
import pytest

from src.infrastructure.handlers.disruptor_ring_buffer import DisruptorRingBuffer


def test_disruptor_ring_buffer_invalid_capacity():
    with pytest.raises(ValueError):
        DisruptorRingBuffer(capacity=100)

    with pytest.raises(ValueError):
        DisruptorRingBuffer(capacity=0)

    with pytest.raises(ValueError):
        DisruptorRingBuffer(capacity=-16)


def test_disruptor_ring_buffer_empty_state():
    buffer = DisruptorRingBuffer(capacity=8)
    assert buffer.capacity == 8
    assert buffer.count == 0
    assert buffer.sequence == 0
    assert buffer.latest_tick() is None

    epochs, prices = buffer.latest_window(5)
    assert len(epochs) == 0
    assert len(prices) == 0

    assert len(buffer.get_prices_array(5)) == 0
    assert len(buffer.get_prices_array(-1)) == 0


def test_disruptor_ring_buffer_push_and_retrieval():
    buffer = DisruptorRingBuffer(capacity=8)
    for i in range(5):
        seq = buffer.push_tick(1000 + i, 100.0 + i)
        assert seq == i

    assert buffer.count == 5
    assert buffer.sequence == 5
    assert buffer.latest_tick() == (1004, 104.0)

    epochs, prices = buffer.latest_window(3)
    np.testing.assert_array_equal(epochs, np.array([1002, 1003, 1004], dtype=np.int64))
    np.testing.assert_array_equal(prices, np.array([102.0, 103.0, 104.0], dtype=np.float64))

    all_prices = buffer.get_prices_array(10)
    np.testing.assert_array_equal(all_prices, np.array([100.0, 101.0, 102.0, 103.0, 104.0], dtype=np.float64))


def test_disruptor_ring_buffer_wrap_around():
    buffer = DisruptorRingBuffer(capacity=4)
    for i in range(10):
        buffer.push_tick(2000 + i, 50.0 + i)

    assert buffer.capacity == 4
    assert buffer.count == 4
    assert buffer.sequence == 10
    assert buffer.latest_tick() == (2009, 59.0)

    epochs, prices = buffer.latest_window(4)
    expected_epochs = np.array([2006, 2007, 2008, 2009], dtype=np.int64)
    expected_prices = np.array([56.0, 57.0, 58.0, 59.0], dtype=np.float64)
    np.testing.assert_array_equal(epochs, expected_epochs)
    np.testing.assert_array_equal(prices, expected_prices)

    prices_2 = buffer.get_prices_array(2)
    np.testing.assert_array_equal(prices_2, np.array([58.0, 59.0], dtype=np.float64))


def test_disruptor_ring_buffer_clear():
    buffer = DisruptorRingBuffer(capacity=4)
    buffer.push_tick(100, 10.0)
    buffer.push_tick(101, 11.0)
    assert buffer.count == 2

    buffer.clear()
    assert buffer.count == 0
    assert buffer.sequence == 0
    assert buffer.latest_tick() is None

    buffer.push_tick(200, 20.0)
    assert buffer.count == 1
    assert buffer.latest_tick() == (200, 20.0)
