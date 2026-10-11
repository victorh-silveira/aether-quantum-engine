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
    assert len(files) == 3
    for path in files:
        dashboard = json.loads(path.read_text())
        assert dashboard["panels"]
        references = []
        for panel in dashboard["panels"]:
            sql = panel["targets"][0]["rawSql"]
            assert sql
            assert "$__timeFilter(to_timestamp(candle_epoch))" not in sql
            assert panel["options"].get("spanNulls") is not True
            references.append(panel.get("datasource"))
            references.extend(target.get("datasource") for target in panel["targets"])
        references.extend(variable.get("datasource") for variable in dashboard.get("templating", {}).get("list", []))
        assert all(
            reference == {"type": "grafana-postgresql-datasource", "uid": "timescale"}
            for reference in references
            if reference is not None
        )
        if path.name == "synthetic_asset.json":
            assert (
                sum(
                    "$__unixEpochFilter(candle_epoch)" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"]
                )
                == 3
            )
            variables = dashboard["templating"]["list"]
            assert all("FROM valid" in variable["query"] for variable in variables)
            assert all("reason='ok'" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"])
            assert all("indicator_model_versions" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"])
            assert all(
                "NULLIF('${period:raw}','')::integer" in panel["targets"][0]["rawSql"]
                and "${period}" not in panel["targets"][0]["rawSql"]
                for panel in dashboard["panels"]
            )
            history = next(
                panel for panel in dashboard["panels"] if panel["title"] == "Historico auditavel de leituras"
            )
            assert "candle_epoch + 2 * NULLIF('${period:raw}','')::integer" in history["targets"][0]["rawSql"]
        elif path.name == "synthetic_overview.json":
            assert all("FROM valid" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"])
            assert all("l.reason='ok'" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"])
            assert all("DISTINCT ON" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"])
            assert all("indicator_model_versions" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"])
        else:
            assert dashboard["uid"] == "aether-synthetic-catalog"
            assert "FROM indicator_symbols WHERE active" in dashboard["panels"][0]["targets"][0]["rawSql"]
            assert "CROSS JOIN (VALUES (300),(900),(3600))" in dashboard["panels"][-1]["targets"][0]["rawSql"]
            assert "modelo_atualizado_aguardando_fechamento" in dashboard["panels"][-1]["targets"][0]["rawSql"]
            assert "s.reason='ok'" in dashboard["panels"][-1]["targets"][0]["rawSql"]
        assert all(
            "reason IN ('ok','experimental')" not in panel["targets"][0]["rawSql"] for panel in dashboard["panels"]
        )


def test_compose_scrapes_running_indicator_service():
    root = Path(__file__).resolve().parents[2]
    compose = (root / "infra/docker/docker-compose.yml").read_text()
    prometheus = (root / "infra/docker/prometheus/prometheus.yml").read_text()
    assert "  indicator:" in compose
    assert "condition: service_healthy" in compose
    assert "  prometheus:\n    depends_on:\n      indicator:\n        condition: service_healthy" in compose
    assert "indicator:9101" in prometheus
    assert "host.docker.internal" not in prometheus
    assert "  grafana:\n" in compose
    assert "      timescaledb:\n        condition: service_healthy" in compose
    assert "127.0.0.1:3001:3000" in compose
    assert "GF_SECURITY_ADMIN_USER: admin" in compose
    assert "GF_SECURITY_ADMIN_PASSWORD: aether" in compose


def test_config_scope_matches_volatility_catalog():
    import json

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "config/settings.json").read_text())
    assert config["catalog_scope"] == "volatility_indices"


def test_make_exposes_main_command_groups_and_default_help():
    import re
    import subprocess

    root = Path(__file__).resolve().parents[2]
    source = (root / "Makefile").read_text()
    targets = set()
    for line in source.splitlines():
        match = re.fullmatch(r"([a-z][a-z0-9_-]*(?: [a-z][a-z0-9_-]*)*):.*", line)
        if match:
            targets.update(match.group(1).split())
    expected = {
        "help",
        "app-install",
        "app-run",
        "app-train",
        "app-test",
        "app-lint",
        "app-security",
        "app-clean",
        "app-pre-commit",
        "app-pre-commit-run",
        "app-setup-wsl",
        "docker-up",
        "docker-down",
        "docker-restart",
        "docker-rebuild",
        "docker-reset",
        "docker-clean",
        "docker-ps",
        "docker-logs",
        "docker-smoke",
        "docker-bash",
        "docker-hydrate",
        "docker-timescale-lifecycle",
        "docker-sanitize-run",
        "docker-sanitize-run-keep-meta",
        "app-rust-lint",
        "app-rust-validate",
        "app-rust-security",
        "app-rust-test",
        "app-rust-build",
        "pre-commit",
        "timescale-lifecycle",
    }
    assert expected <= targets
    result = subprocess.run(["make", "--no-print-directory"], cwd=root, check=True, capture_output=True, text=True)
    assert "MENU DE AJUDA" in result.stdout
    assert "app-train" in result.stdout and "docker-up" in result.stdout


def test_ci_validates_indicator_before_main_release():
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    assert "  indicator:\n" in workflow
    assert "make indicator-lint indicator-test" in workflow
    assert "needs: [indicator, python, docker, shell, rust, workflows]" in workflow
    assert '      - "indicator/**"' in workflow
