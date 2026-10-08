"""Sincronizador continuo de relogio NTP de baixa latencia com a API Deriv."""

from __future__ import annotations

import time


class DerivNtpSynchronizer:
    """Calcula e mantem o deslocamento temporal atomico com os servidores da Deriv."""

    def __init__(self, *, max_samples: int = 20) -> None:
        """Inicializa historico de offsets e estimativa de latencia de ida e volta."""
        self._max_samples = max(1, int(max_samples))
        self._offset_seconds: float = 0.0
        self._estimated_rtt: float = 0.05
        self._is_synchronized: bool = False
        self._samples: list[float] = []

    @property
    def is_synchronized(self) -> bool:
        """Indica se ao menos uma sincronizacao bem sucedida foi realizada."""
        return self._is_synchronized

    @property
    def clock_offset_seconds(self) -> float:
        """Retorna a diferenca estimada em segundos entre o servidor e o host local."""
        return self._offset_seconds

    def update_from_server_time(
        self,
        request_sent_local_epoch: float,
        server_reported_epoch: float,
        response_received_local_epoch: float,
    ) -> float:
        """Processa a mensagem de resposta temporal ajustando o offset por metade do RTT."""
        t0 = float(request_sent_local_epoch)
        t1 = float(server_reported_epoch)
        t2 = float(response_received_local_epoch)

        rtt = max(0.0, t2 - t0)
        corrected_server_epoch = t1 + (rtt / 2.0)
        local_midpoint = (t0 + t2) / 2.0
        offset = corrected_server_epoch - local_midpoint

        self._samples.append(offset)
        if len(self._samples) > self._max_samples:
            self._samples.pop(0)

        self._offset_seconds = sum(self._samples) / len(self._samples)
        self._estimated_rtt = rtt
        self._is_synchronized = True

        return self._offset_seconds

    def synchronized_epoch(self) -> float:
        """Retorna o timestamp corrente do broker corrigido pelo offset atomico."""
        now = time.time()
        return now + self._offset_seconds if self._is_synchronized else now
