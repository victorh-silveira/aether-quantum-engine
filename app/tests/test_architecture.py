"""Prova de que o processo ativo nao possui rotas de negociacao."""

import ast
from pathlib import Path


def test_active_code_has_no_buy_or_authorization_route():
    root = Path(__file__).resolve().parents[1]
    active = [root / "run.py", root / "train.py", *sorted((root / "src" / "indicator").glob("*.py"))]
    banned = {"buy", "proposal", "authorize", "sell", "portfolio", "balance", "contract_update"}
    for path in active:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        assert key.value not in banned, (path, key.value)
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                assert not any("trade_handler" in name or "execution_orders" in name for name in names)


def test_dashboards_have_real_queries_and_no_interpolation():
    import json

    root = Path(__file__).resolve().parents[2]
    files = list((root / "infra/docker/grafana/provisioning/dashboards").glob("synthetic_*.json"))
    assert len(files) == 2
    for path in files:
        dashboard = json.loads(path.read_text())
        assert dashboard["panels"]
        for panel in dashboard["panels"]:
            assert panel["targets"][0]["rawSql"]
            assert panel["options"].get("spanNulls") is not True


def test_compose_scrapes_running_indicator_service():
    root = Path(__file__).resolve().parents[2]
    compose = (root / "infra/docker/docker-compose.yml").read_text()
    prometheus = (root / "infra/docker/prometheus/prometheus.yml").read_text()
    assert "  indicator:" in compose
    assert "condition: service_healthy" in compose
    assert "  prometheus:\n    depends_on:\n      indicator:\n        condition: service_healthy" in compose
    assert "indicator:9101" in prometheus
    assert "host.docker.internal" not in prometheus
