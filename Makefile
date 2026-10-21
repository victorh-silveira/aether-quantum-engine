# ==============================================================================
#                 AETHER QUANTUM ENGINE - CORE MATRIX MAKEFILE
# ==============================================================================

SHELL := /bin/bash
APP_DIR=app
CONDA_ENV ?= deriv-api
DOCKER_DIR=infra/docker
DOCKER_PROFILES ?= core,ml,telemetry,indicator
export COMPOSE_PROFILES := $(DOCKER_PROFILES)
export AETHER_METRICS_HOST_IP ?= $(shell ip -4 route show default 2>/dev/null | awk 'NR==1 {print $$3}')
DOCKER_COMPOSE=docker compose -f $(DOCKER_DIR)/docker-compose.yml --project-directory $(DOCKER_DIR) --env-file .env
DOCKER_LOGS_TAIL ?= 200
DOCKER_LOGS_SERVICES ?= aether-core-cache-redis aether-core-data-timescaledb aether-core-storage-minio aether-ml-meta-classifier aether-ml-loss-classifier aether-market-signal-indicator aether-observe-metrics-prometheus aether-observe-dashboards-grafana

define docker_service_name
$(strip $(or \
	$(if $(filter redis aether-core-cache-redis,$(1)),aether-core-cache-redis),\
	$(if $(filter ts timescale timescaledb aether-core-data-timescaledb,$(1)),aether-core-data-timescaledb),\
	$(if $(filter minio aether-core-storage-minio,$(1)),aether-core-storage-minio),\
	$(if $(filter meta meta-classifier aether-ml-meta-classifier,$(1)),aether-ml-meta-classifier),\
	$(if $(filter loss loss-classifier aether-ml-loss-classifier,$(1)),aether-ml-loss-classifier),\
	$(if $(filter indicator aether-market-signal-indicator,$(1)),aether-market-signal-indicator),\
	$(if $(filter prom prometheus aether-observe-metrics-prometheus,$(1)),aether-observe-metrics-prometheus),\
	$(if $(filter grafana aether-observe-dashboards-grafana,$(1)),aether-observe-dashboards-grafana),\
	$(1)))
endef

RESOLVE_PY := $(shell bash linters/git-hooks/bin/resolve_conda_python.sh 2>/dev/null || echo python)
PYTHON := $(RESOLVE_PY)
INDICATOR_PYTHON ?= $(if $(wildcard indicator/.venv/bin/python),$(abspath indicator/.venv/bin/python),$(if $(wildcard .venv/bin/python),$(abspath .venv/bin/python),$(PYTHON)))
INDICATOR_TRAIN_MEMORY_MB ?= 3072
QA_PYTHON ?= $(INDICATOR_PYTHON)

GREEN  := \033[1;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
CYAN   := \033[1;36m
RED    := \033[1;31m
RESET  := \033[0m

.DEFAULT_GOAL := help

.PHONY: app-run app-train app-test app-lint app-workflows-lint app-validate app-security app-build app-clean app-install \
	indicator-up indicator-down indicator-train indicator-test indicator-lint indicator-security app-train-all \
	app-pre-commit app-pre-commit-run app-setup-wsl help docker-up docker-down \
	docker-restart docker-rebuild docker-reset docker-clean docker-ps docker-logs \
	docker-smoke docker-bash docker-hydrate docker-timescale-lifecycle lint test


help:
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "$(GREEN)                   AETHER QUANTUM ENGINE - MENU DE AJUDA                $(RESET)"
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "Uso: $(CYAN)make <comando>$(RESET)"
	@echo -e ""
	@echo -e "$(YELLOW)Python:$(RESET) Conda $(CONDA_ENV) ($(PYTHON))"
	@echo -e ""
	@echo -e "$(YELLOW)App:$(RESET)"
	@echo -e "  $(GREEN)app-run$(RESET)            - Sobe o motor em execucao real/demo (run.py)"
	@echo -e "  $(GREEN)app-train$(RESET)          - Sobe a stack e diagnostica setup M5/M15/H1"
	@echo -e "  $(GREEN)app-research-ml$(RESET)    - Pesquisa legada TCN, meta e loss separada da execucao"
	@echo -e "  $(GREEN)app-lint$(RESET)           - Lint Python/JSON/YAML/Docker/Shell/Rust/Workflows + indicador"
	@echo -e "  $(GREEN)app-validate$(RESET)       - Valida codigo, configs, Compose, Shell e Rust"
	@echo -e "  $(GREEN)app-security$(RESET)       - Seguranca Python/Docker/Rust + indicador"
	@echo -e "  $(GREEN)app-test$(RESET)           - Testes Python/JSON/Docker/Shell/Rust + indicador; cobertura 100%"
	@echo -e "  $(GREEN)app-build$(RESET)          - Build/compilacao da matriz local do CI"
	@echo -e "  $(GREEN)app-clean$(RESET)          - Limpa caches, artefactos e logs locais"
	@echo -e "  $(GREEN)app-install$(RESET)        - Instala dependencias no Conda $(CONDA_ENV)"
	@echo -e "  $(GREEN)app-pre-commit$(RESET)     - Instala e valida githooks pre-commit"
	@echo -e "  $(GREEN)app-pre-commit-run$(RESET) - Executa a matriz completa de hooks em todos os arquivos"
	@echo -e ""
	@echo -e "$(YELLOW)Docker:$(RESET)"
	@echo -e "  $(GREEN)docker-up$(RESET)          - Sobe motor/ML + indicador + telemetria em um Compose"
	@echo -e "  $(GREEN)docker-rebuild$(RESET)     - Rebuilda meta/loss/indicador e recarrega modelos locais"
	@echo -e "  $(GREEN)docker-reset$(RESET)       - $(RED)DESTRUTIVO$(RESET): sanitiza run + loss-models + volumes, bootstrap e sobe stack"
	@echo -e "  $(GREEN)docker-clean$(RESET)       - $(RED)DESTRUTIVO$(RESET): para e remove containers, redes e volumes da stack"
	@echo -e "  $(GREEN)docker-down$(RESET)        - Para os containers da stack (preserva dados e volumes)"
	@echo -e "  $(GREEN)docker-restart$(RESET)     - Reinicia os containers da stack (preserva dados)"
	@echo -e "  $(GREEN)docker-ps$(RESET)          - Exibe o status atual dos containers"
	@echo -e "  $(GREEN)docker-logs$(RESET)        - Logs dos servicos running (tail=200; DOCKER_SERVICE=minio-init|... F=1)"
	@echo -e "  $(GREEN)docker-smoke$(RESET)       - Executa smoke checks e verificacao de saude da stack"
	@echo -e "$(BLUE)========================================================================$(RESET)"

# ------------------------------------------------------------------------------
# App Targets
# ------------------------------------------------------------------------------

app-install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(APP_DIR)/requirements.txt -r $(APP_DIR)/requirements-dev.txt
	$(MAKE) -C indicator app-install

app-lint:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage lint
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage lint --config-text json
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage lint --config-text yaml
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area docker --stage lint
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area shell --stage lint
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage lint
	@$(MAKE) --no-print-directory app-workflows-lint

app-workflows-lint:
	@if command -v actionlint >/dev/null 2>&1; then actionlint .github/workflows/ci.yml; else echo '[workflows] actionlint ausente; job CI - Workflows valida no GitHub'; fi

app-validate:
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage validate
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage validate --config-text json
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage validate --config-text yaml
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area docker --stage validate
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area shell --stage validate
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage validate

app-test:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage test --coverage-fail-under 100
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage test --config-text json
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area docker --stage test
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area shell --stage test
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage test

app-security:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage security
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area docker --stage security
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area shell --stage security
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage security

app-build:
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area python --stage build
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area docker --stage build
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area shell --stage build
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage build

app-rust-lint:
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage lint

app-rust-validate:
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage validate

app-rust-security:
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage security

app-rust-test:
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage test

app-rust-build:
	$(QA_PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --area rust --stage build

app-clean:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage clean
	$(MAKE) -C indicator app-clean

app-run:
	$(PYTHON) run.py

app-train:
	@$(MAKE) --no-print-directory docker-up
	@$(MAKE) --no-print-directory indicator-train

app-research-ml:
	@cd $(APP_DIR) && $(PYTHON) -m scripts.operations.train_loss_classifier
	$(PYTHON) $(APP_DIR)/train.py
	$(PYTHON) $(APP_DIR)/scripts/operations/check_dl_checkpoint.py
	@$(PYTHON) $(APP_DIR)/scripts/operations/ensure_timescale.py || echo '[AVISO] Semente Timescale indisponivel; meta tentara API Deriv.'
	$(PYTHON) $(APP_DIR)/scripts/operations/train_meta_classifier.py --trials 60 --bars 5000 --source auto --candidate-on-low-quality --export-min-zscore -0.05 --export-min-ir -0.50
	$(DOCKER_COMPOSE) up -d --no-deps --force-recreate aether-ml-meta-classifier aether-ml-loss-classifier
	@bash infra/docker/sh/docker-wait-healthy.sh
	@bash infra/docker/sh/docker-smoke.sh

indicator-up: docker-up

indicator-down: docker-down

indicator-train:
	@if [ "$$($(INDICATOR_PYTHON) -c 'import sys; print(sys.platform)')" = linux ] && command -v prlimit >/dev/null 2>&1; then \
		OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
		prlimit --as=$$(( $(INDICATOR_TRAIN_MEMORY_MB) * 1024 * 1024 )) -- $(INDICATOR_PYTHON) indicator/app/train.py; \
	else \
		OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
		$(INDICATOR_PYTHON) indicator/app/train.py; \
	fi

indicator-test:
	$(MAKE) -C indicator test PYTHON=$(INDICATOR_PYTHON)

indicator-lint:
	$(MAKE) -C indicator lint PYTHON=$(INDICATOR_PYTHON)

indicator-security:
	$(MAKE) -C indicator app-security PYTHON=$(INDICATOR_PYTHON)

app-train-all: app-train

app-pre-commit:
	bash linters/git-hooks/install.sh
	chmod +x linters/git-hooks/bin/resolve_conda_python.sh linters/git-hooks/bin/python

app-pre-commit-run:
	$(QA_PYTHON) -m pre_commit run --all-files -c .pre-commit-config.yaml

app-setup-wsl:
	bash app/scripts/wsl/setup.sh

# ------------------------------------------------------------------------------
# Docker Targets
# ------------------------------------------------------------------------------

docker-sanitize-run:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-sanitize-run · limpa checkpoints e artefactos da run anterior"'
	$(PYTHON) $(APP_DIR)/scripts/operations/sanitize_fresh_run.py

docker-sanitize-run-keep-meta:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-sanitize-run-keep-meta · limpa TCN/loss/estado (mantem meta_lgbm.pkl)"'
	$(PYTHON) $(APP_DIR)/scripts/operations/sanitize_fresh_run.py --keep-meta-bundle

docker-up:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-up · Aether stack (profiles: $(DOCKER_PROFILES))"'
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 1 5 "Host prerequisites"'
	@bash infra/docker/sh/host-prereq.sh
	@test -f .env || cp .env.example .env
	@mkdir -p indicator/data/indicator-models
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 2 5 "Compose up"'
	$(DOCKER_COMPOSE) up -d --build aether-market-signal-indicator
	$(DOCKER_COMPOSE) up -d
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 3 5 "Healthchecks"'
	@bash infra/docker/sh/docker-wait-healthy.sh
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 4 5 "Timescale lifecycle + hydrate"'
	@bash infra/docker/sh/timescale-lifecycle.sh
	@bash infra/docker/sh/docker-hydrate.sh
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 5 5 "Smoke checks"'
	@bash infra/docker/sh/docker-smoke.sh

docker-rebuild:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-rebuild · rebuild meta/loss/indicador (preserva modelos locais)"'
	@test -f .env || cp .env.example .env
	@if [ ! -f $(DOCKER_DIR)/loss-models/loss_bootstrap_live64.pkl ]; then \
		cd $(APP_DIR) && LOKY_MAX_CPU_COUNT=$${LOKY_MAX_CPU_COUNT:-4} $(PYTHON) -m scripts.operations.train_loss_classifier; \
	fi
	$(DOCKER_COMPOSE) build --pull aether-ml-meta-classifier aether-ml-loss-classifier aether-market-signal-indicator
	$(DOCKER_COMPOSE) up -d --force-recreate aether-ml-meta-classifier aether-ml-loss-classifier aether-market-signal-indicator
	$(DOCKER_COMPOSE) up -d
	@bash infra/docker/sh/docker-wait-healthy.sh
	@bash infra/docker/sh/timescale-lifecycle.sh
	@bash infra/docker/sh/docker-smoke.sh

docker-timescale-lifecycle:
	@bash infra/docker/sh/timescale-lifecycle.sh

docker-down:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-down · parando stack (volumes preservados)"'
	$(DOCKER_COMPOSE) down
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_nl'

docker-restart:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-restart · reiniciando containers (volumes preservados)"'
	@test -f .env || cp .env.example .env
	$(DOCKER_COMPOSE) restart
	@bash infra/docker/sh/docker-wait-healthy.sh
	@$(DOCKER_COMPOSE) ps

docker-reset:
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-reset · ATENCAO: sanitiza run + loss-models + volumes"'
	@echo -e "$(RED)  Limpando checkpoints/artefactos + loss-models + volumes Redis/Timescale/MinIO; recria a stack$(RESET)"
	@echo ""
	@$(MAKE) --no-print-directory docker-sanitize-run-keep-meta
	@bash infra/docker/sh/loss-clf-reset.sh clear
	@cd $(APP_DIR) && LOKY_MAX_CPU_COUNT=$${LOKY_MAX_CPU_COUNT:-4} $(PYTHON) -m scripts.operations.train_loss_classifier
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@$(MAKE) --no-print-directory docker-up

docker-clean:
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-clean · ATENCAO: volumes serao apagados"'
	@echo -e "$(RED)  Removendo containers, redes e volumes (TimescaleDB/MinIO/Redis)$(RESET)"
	@echo ""
	@bash infra/docker/sh/loss-clf-reset.sh clear
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@echo ""
	@$(DOCKER_COMPOSE) ps

docker-hydrate:
	@bash infra/docker/sh/docker-hydrate.sh

docker-smoke:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-smoke"'
	@bash infra/docker/sh/docker-smoke.sh

docker-ps:
	$(DOCKER_COMPOSE) ps

docker-logs:
	$(DOCKER_COMPOSE) logs --tail=$(DOCKER_LOGS_TAIL) $(if $(F),-f,) $(if $(DOCKER_SERVICE),$(call docker_service_name,$(DOCKER_SERVICE)),$(DOCKER_LOGS_SERVICES))

docker-bash:
	$(DOCKER_COMPOSE) exec -it $(call docker_service_name,$(or $(DOCKER_SERVICE),aether-core-data-timescaledb)) sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'

# ------------------------------------------------------------------------------
# Aliases de Compatibilidade
# ------------------------------------------------------------------------------
pre-commit: app-pre-commit
timescale-lifecycle: docker-timescale-lifecycle
lint: app-lint
test: app-test
