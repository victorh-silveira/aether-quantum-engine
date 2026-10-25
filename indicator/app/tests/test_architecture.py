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
            assert panel["type"] == "table"
            assert "indicator_model_registry" not in sql
            references.append(panel.get("datasource"))
            references.extend(target.get("datasource") for target in panel["targets"])
        references.extend(variable.get("datasource") for variable in dashboard.get("templating", {}).get("list", []))
        assert all(
            reference == {"type": "grafana-postgresql-datasource", "uid": "timescale"}
            for reference in references
            if reference is not None
        )
        if path.name == "synthetic_asset.json":
            assert any(
                "$__unixEpochFilter(candle_epoch)" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"]
            )
            variables = dashboard["templating"]["list"]
            assert [variable["name"] for variable in variables] == ["symbol", "period"]
            assert all("${symbol:sqlstring}" in panel["targets"][0]["rawSql"] for panel in dashboard["panels"])
        elif path.name == "synthetic_overview.json":
            assert "pullback_signals" in dashboard["panels"][0]["targets"][0]["rawSql"]
            assert "pullback_broker_status" in dashboard["panels"][1]["targets"][0]["rawSql"]
        else:
            assert dashboard["uid"] == "aether-synthetic-catalog"
            assert "CROSS JOIN (VALUES (300),(900),(3600))" in dashboard["panels"][0]["targets"][0]["rawSql"]
            assert "dados_atrasados" in dashboard["panels"][0]["targets"][0]["rawSql"]
        assert all(
            "reason IN ('ok','experimental')" not in panel["targets"][0]["rawSql"] for panel in dashboard["panels"]
        )


def test_compose_scrapes_running_indicator_service():
    root = Path(__file__).resolve().parents[3]
    compose = (root / "infra/docker/docker-compose.yml").read_text()
    prometheus = (root / "infra/docker/prometheus/prometheus.yml").read_text()
    provisioning = (root / "infra/docker/grafana/provisioning/dashboards/dashboards.yml").read_text()
    assert "  aether-market-signal-indicator:" in compose
    assert "condition: service_healthy" in compose
    assert "profiles: [indicator]" in compose
    assert "@aether-core-data-timescaledb:5432/" in compose
    assert "aether-market-signal-indicator:9101" in prometheus
    assert "host.docker.internal:9100" in prometheus
    assert "  aether-observe-dashboards-grafana:\n" in compose
    assert "127.0.0.1:3000:3000" in compose
    assert "path: /etc/grafana/indicator-dashboards" in provisioning
    assert not (root / "indicator/infra/docker/docker-compose.yml").exists()


def test_root_make_starts_unified_stack_and_trains_in_order():
    root = Path(__file__).resolve().parents[3]
    makefile = (root / "Makefile").read_text()
    training = makefile.split("\napp-train:\n", 1)[1].split("\napp-research-ml:", 1)[0]
    assert "docker-up" in training and "indicator-train" in training
    assert "train_loss_classifier" not in training
    research = makefile.split("\napp-research-ml:\n", 1)[1].split("\nindicator-up:", 1)[0]
    assert "train.py" in research and "check_dl_checkpoint.py" in research
    assert "train_loss_classifier" not in research and "train_meta_classifier" not in research
    assert "app-train-all: app-train" in makefile
    assert "lint: app-lint" in makefile
    assert "test: app-test" in makefile
    lint = makefile.split("\napp-lint:\n", 1)[1].split("\napp-validate:\n", 1)[0]
    test = makefile.split("\napp-test:\n", 1)[1].split("\napp-security:\n", 1)[0]
    lint_stages = (
        "--area python --stage lint",
        "--config-text json",
        "--config-text yaml",
        "--area docker --stage lint",
        "--area shell --stage lint",
        "--area rust --stage lint",
        "app-workflows-lint",
    )
    for stage in lint_stages:
        assert stage in lint
    test_stages = (
        "--area python --stage test --coverage-fail-under 100",
        "--config-text json",
        "--area docker --stage test",
        "--area shell --stage test",
        "--area rust --stage test",
    )
    for stage in test_stages:
        assert stage in test
    hooks = (root / ".pre-commit-config.yaml").read_text()
    assert "id: python-lint" in hooks
    assert "id: python-security" in hooks
    assert "id: python-test" in hooks
    assert "id: workflows-lint" in hooks
    assert (
        "resolve_dl_train_symbols" in (root / "app/src/application/services/orchestrator/config_symbols.py").read_text()
    )
    assert "$(MAKE) -C .. app-train" in (root / "indicator/Makefile").read_text()
    assert "DOCKER_PROFILES ?= core,telemetry,indicator" in makefile
    assert "127.0.0.1:5432/aether" in (root / "indicator/config/settings.json").read_text()


def test_config_scope_matches_volatility_catalog():
    import json

    root = Path(__file__).resolve().parents[2]
    config = json.loads((root / "config/settings.json").read_text())
    assert config["catalog_scope"] in ("volatility_indices", "stock_indices")


def test_make_exposes_main_command_groups_and_default_help():
    import re
    import shutil
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
    if shutil.which("make") is None:
        return
    result = subprocess.run(["make", "--no-print-directory"], cwd=root, check=True, capture_output=True, text=True)
    assert "MENU DE AJUDA" in result.stdout
    assert "app-train" in result.stdout and "docker-up" in result.stdout


def test_ci_validates_indicator_before_main_release():
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    assert "--area python --stage lint" in workflow
    assert "--area python --stage security" in workflow
    assert "--area python --stage test" in workflow
    clean_workspace = (root / "app/scripts/operations/clean_workspace.py").read_text()
    assert "_run_indicator_lint" in clean_workspace
    assert "_run_indicator_security" in clean_workspace
    assert "_run_indicator_test" in clean_workspace
    assert "needs: [python, docker, shell, rust, workflows]" in workflow
    assert '      - "indicator/**"' in workflow
