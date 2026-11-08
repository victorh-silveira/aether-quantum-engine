"""Contrato de bind de metricas para scrape Docker no host Windows."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.application.services.orchestrator import Orchestrator
from src.domain.config_knobs import load_settings_json


def test_orchestrator_uses_telemetry_host_and_port_from_settings(orch_config):
    orch_config["telemetry"] = load_settings_json()["telemetry"]
    with patch("src.application.services.orchestrator.WebSocketManager", return_value=AsyncMock()) as ws_class:
        ws_class.return_value.subscribe = MagicMock()
        orch = Orchestrator(orch_config, "token")
    assert orch.metrics_server._host == orch_config["telemetry"]["host"]
    assert orch.metrics_server._port == 9100
