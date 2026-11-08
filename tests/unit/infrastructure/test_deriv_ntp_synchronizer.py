"""Testes unitarios para o sincronizador NTP com a Deriv."""

import pytest

from src.infrastructure.websocket.deriv_ntp_synchronizer import DerivNtpSynchronizer


def test_deriv_ntp_synchronizer_initial_state():
    sync = DerivNtpSynchronizer()
    assert not sync.is_synchronized
    assert sync.clock_offset_seconds == 0.0

    epoch = sync.synchronized_epoch()
    assert isinstance(epoch, float)
    assert epoch > 0.0


def test_deriv_ntp_synchronizer_offset_calculation():
    sync = DerivNtpSynchronizer()

    offset = sync.update_from_server_time(
        request_sent_local_epoch=100.0,
        server_reported_epoch=100.5,
        response_received_local_epoch=100.1,
    )

    assert sync.is_synchronized
    assert offset == pytest.approx(0.5, abs=0.01)
    assert sync.clock_offset_seconds == pytest.approx(0.5, abs=0.01)


def test_deriv_ntp_synchronizer_rolling_samples():
    sync = DerivNtpSynchronizer(max_samples=2)
    sync.update_from_server_time(10.0, 10.4, 10.2)
    sync.update_from_server_time(20.0, 20.6, 20.2)
    sync.update_from_server_time(30.0, 30.6, 30.2)

    assert sync.is_synchronized
    assert len(sync._samples) == 2
