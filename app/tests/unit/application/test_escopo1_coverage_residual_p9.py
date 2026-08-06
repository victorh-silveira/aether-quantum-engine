"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

import json
from io import StringIO
from types import SimpleNamespace


class _FakePath:
    def __init__(self, payload):
        self._payload = payload

    def open(self, *_a, **_k):
        text = self._payload if isinstance(self._payload, str) else json.dumps(self._payload)
        return StringIO(text)


def test_side_equilibrium_redis_without_hset():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {}}},
        state_store=SimpleNamespace(client=object()),
        timescale_writer=None,
    )
    from src.application.services.side_equilibrium_store import record_side_equilibrium_outcome

    record_side_equilibrium_outcome(orch, "OTC_SPC", direction="CALL", won=True)
