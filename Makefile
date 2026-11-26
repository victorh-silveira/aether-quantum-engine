# ==============================================================================
#                 AETHER QUANTUM ENGINE - CORE MATRIX MAKEFILE
# ==============================================================================

SHELL := /bin/bash
APP_DIR=.
CONDA_ENV ?= deriv-api
DOCKER_DIR=infra/docker
DOCKER_PROFILES ?= core,telemetry
export COMPOSE_PROFILES := $(DOCKER_PROFILES)
export AETHER_METRICS_HOST_IP ?= $(shell ip -4 route show default 2>/dev/null | awk 'NR==1 {print $$3}')
export OTEL_EXPORTER_OTLP_ENDPOINT ?= http://127.0.0.1:4318
export WSLENV := $(if $(WSLENV),$(WSLENV):,)OTEL_EXPORTER_OTLP_ENDPOINT
DOCKER_COMPOSE=docker compose -f $(DOCKER_DIR)/docker-compose.yml --project-directory $(DOCKER_DIR) --env-file .env
DOCKER_LOGS_TAIL ?= 200
DOCKER_LOGS_SERVICES ?= aether-core-cache-redis aether-core-data-timescaledb aether-core-storage-minio aether-observe-metrics-prometheus aether-observe-dashboards-grafana

define docker_service_name
$(strip $(or \
	$(if $(filter redis aether-core-cache-redis,$(1)),aether-core-cache-redis),\
	$(if $(filter ts timescale timescaledb aether-core-data-timescaledb,$(1)),aether-core-data-timescaledb),\
	$(if $(filter minio aether-core-storage-minio,$(1)),aether-core-storage-minio),\
	$(if $(filter prom prometheus aether-observe-metrics-prometheus,$(1)),aether-observe-metrics-prometheus),\
	$(if $(filter grafana aether-observe-dashboards-grafana,$(1)),aether-observe-dashboards-grafana),\
	$(1)))
endef

RESOLVE_PY := $(shell bash linters/git-hooks/bin/resolve_conda_python.sh 2>/dev/null || echo python)
PYTHON := $(RESOLVE_PY)
INDICATOR_PYTHON ?= $(PYTHON)
INDICATOR_TRAIN_MEMORY_MB ?= 3072
QA_PYTHON ?= $(INDICATOR_PYTHON)
LAYA_DATA ?= .cache/laya-r10-gpu-20261126
LAYA_RUN ?= .cache/laya-r10-gpu-run
LAYA_SETTINGS ?= config/settings.json
LAYA_DEVICE ?= cuda
LAYA_TRAIN_PROFILE ?= auto
LAYA_ARGS ?=

GREEN  := \033[1;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
CYAN   := \033[1;36m
RED    := \033[1;31m
RESET  := \033[0m

.DEFAULT_GOAL := help

.PHONY: app-run app-test app-lint app-workflows-lint app-validate app-security app-build app-clean app-install \
	app-prepare app-train app-train-dry-run app-evaluate app-promote app-cuda-install app-cuda-check app-cuda-runtime-check app-cuda-train-check \
	app-pre-commit app-pre-commit-run app-commit app-setup-wsl help docker-up docker-down \
	docker-restart docker-reset docker-clean docker-ps docker-logs \
	docker-smoke docker-bash docker-hydrate docker-timescale-lifecycle lint test


define help_command
@printf '    $(GREEN)%-28s$(RESET) %s\n' '$(1)' '$(2)'
endef

define help_subcategory
@printf '\n  $(CYAN)%s$(RESET)\n' '$(1)'
endef

help:
	@printf '$(BLUE)========================================================================$(RESET)\n'
	@printf '$(GREEN)AETHER QUANTUM ENGINE - MENU DE AJUDA$(RESET)\n'
	@printf '$(BLUE)========================================================================$(RESET)\n'
	@printf '\n$(YELLOW)APP$(RESET)\n'
	$(call help_command,app-install,Instala dependencias do projeto no ambiente Conda)
	$(call help_command,app-cuda-install,Instala PyTorch com CUDA e verifica o acesso a GPU)
	$(call help_command,app-cuda-check,Executa inferencia real do Laya na GPU sem enviar ordens)
	$(call help_command,app-prepare,Prepara 90 dias de velas R_10 M5 e separa as particoes)
	$(call help_command,app-train-dry-run,Valida o dataset e o limite de tokens sem carregar pesos)
	$(call help_command,app-train,Treina o Laya na GPU e seleciona o melhor candidato)
	$(call help_command,app-evaluate,Compara o candidato com o modelo base no teste reservado)
	$(call help_command,app-promote,Aprova o checkpoint validado sem ativar sua execucao)
	$(call help_command,app-run,Verifica CUDA e inicia o motor com o checkpoint configurado)
	@printf '\n$(YELLOW)CI/CD$(RESET)\n'
	$(call help_subcategory,Local)
	$(call help_command,app-lint,Verifica estilo e qualidade de codigo e configuracoes)
	$(call help_command,app-test,Executa a suite de testes com cobertura obrigatoria de 100%)
	$(call help_command,app-security,Audita dependencias e verifica vulnerabilidades e segredos)
	$(call help_subcategory,Pre-commit)
	$(call help_command,app-pre-commit,Instala os hooks Git de validacao e padrao de commit)
	$(call help_command,app-pre-commit-run,Executa todos os hooks de qualidade e seguranca do projeto)
	$(call help_command,app-commit,Cria commit Conventional Commits em PT-BR com os hooks ativos)
	$(call help_subcategory,Workflows)
	$(call help_command,app-workflows-lint,Valida a sintaxe e a estrutura dos workflows do GitHub Actions)
	@printf '\n$(YELLOW)DOCKER$(RESET)\n'
	$(call help_command,docker-up,Sobe a stack e verifica a saude dos servicos)
	$(call help_command,docker-down,Para os containers preservando os dados e volumes)
	$(call help_command,docker-restart,Reinicia os servicos e aguarda os healthchecks)
	$(call help_command,docker-ps,Exibe o estado atual dos containers da stack)
	$(call help_command,docker-logs,Exibe os logs dos servicos com filtro por DOCKER_SERVICE)
	$(call help_command,docker-reset,DESTRUTIVO - apaga dados e artefatos e recria a stack)
	$(call help_command,docker-clean,DESTRUTIVO - remove containers e volumes persistidos)
	@printf '$(BLUE)========================================================================$(RESET)\n'

# ------------------------------------------------------------------------------
# App Targets
# ------------------------------------------------------------------------------

app-install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt -r requirements-dev.txt

app-lint:
	$(PYTHON) scripts/operations/clean_workspace.py --area python --stage lint
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage lint --config-text json
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage lint --config-text yaml
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area docker --stage lint
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area shell --stage lint
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage lint
	@$(MAKE) --no-print-directory app-workflows-lint

app-workflows-lint:
	@bash linters/git-hooks/bin/workflows-lint.sh

app-validate:
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage validate
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage validate --config-text json
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage validate --config-text yaml
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area docker --stage validate
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area shell --stage validate
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage validate

app-test:
	$(PYTHON) scripts/operations/clean_workspace.py --area python --stage test --coverage-fail-under 100
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage test --config-text json
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage test --config-text yaml
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area docker --stage test
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area shell --stage test
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage test

app-security:
	$(PYTHON) scripts/operations/clean_workspace.py --area python --stage security
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area docker --stage security
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area shell --stage security
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage security

app-build:
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area python --stage build
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area docker --stage build
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area shell --stage build
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage build

app-rust-lint:
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage lint

app-rust-validate:
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage validate

app-rust-security:
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage security

app-rust-test:
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage test

app-rust-build:
	$(QA_PYTHON) scripts/operations/clean_workspace.py --area rust --stage build

app-clean:
	$(PYTHON) scripts/operations/clean_workspace.py --stage clean

app-cuda-install:
	$(PYTHON) -m pip install --upgrade -r requirements-cuda.txt
	$(PYTHON) -m pip check
	@$(MAKE) --no-print-directory app-cuda-runtime-check

app-cuda-runtime-check:
	$(PYTHON) -m tools.laya_cuda_check --mode runtime --settings "$(LAYA_SETTINGS)"

app-cuda-check:
	$(PYTHON) -m tools.laya_cuda_check --mode laya --settings "$(LAYA_SETTINGS)"

app-cuda-train-check:
	$(PYTHON) -m tools.laya_cuda_check --mode training --settings "$(LAYA_SETTINGS)"

app-prepare:
	$(PYTHON) -m tools.laya_workflow prepare --data "$(LAYA_DATA)" --settings "$(LAYA_SETTINGS)" $(LAYA_ARGS)

app-train: app-cuda-runtime-check
	$(PYTHON) -m tools.laya_workflow train --data "$(LAYA_DATA)" --run "$(LAYA_RUN)" --profile "$(LAYA_TRAIN_PROFILE)" $(LAYA_ARGS)

app-train-dry-run:
	$(PYTHON) -m tools.laya_workflow train --data "$(LAYA_DATA)" --run "$(LAYA_RUN)" --dry-run $(LAYA_ARGS)

app-evaluate:
	$(PYTHON) -m tools.laya_workflow evaluate --data "$(LAYA_DATA)" --run "$(LAYA_RUN)" --device "$(LAYA_DEVICE)" $(LAYA_ARGS)

app-promote:
	$(PYTHON) -m tools.laya_workflow promote --data "$(LAYA_DATA)" --run "$(LAYA_RUN)" $(LAYA_ARGS)

app-run: app-cuda-runtime-check
	$(PYTHON) run.py

app-pre-commit:
	bash linters/git-hooks/install.sh
	chmod +x linters/git-hooks/bin/resolve_conda_python.sh linters/git-hooks/bin/python

app-pre-commit-run:
	$(QA_PYTHON) -m pre_commit run --all-files -c .pre-commit-config.yaml

app-commit:
	@bash scripts/wsl/commit.sh $(ARGS)

app-setup-wsl:
	bash scripts/wsl/setup.sh

# ------------------------------------------------------------------------------
# Docker Targets
# ------------------------------------------------------------------------------

docker-sanitize-run:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-sanitize-run · limpa checkpoints e artefactos da run anterior"'
	$(PYTHON) scripts/operations/sanitize_fresh_run.py

docker-sanitize-run-keep-meta:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-sanitize-run-keep-meta · limpa TCN/loss/estado (mantem meta_lgbm.pkl)"'
	$(PYTHON) scripts/operations/sanitize_fresh_run.py --keep-meta-bundle

docker-up:
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-up · Aether stack (profiles: $(DOCKER_PROFILES))"'
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 1 5 "Host prerequisites"'
	@bash infra/docker/sh/host-prereq.sh
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 2 5 "Compose up"'
	$(DOCKER_COMPOSE) up -d --build
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 3 5 "Healthchecks"'
	@bash infra/docker/sh/docker-wait-healthy.sh
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 4 5 "Timescale lifecycle + hydrate"'
	@bash infra/docker/sh/timescale-lifecycle.sh
	@bash infra/docker/sh/docker-hydrate.sh
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_step 5 5 "Smoke checks"'
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
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-reset · ATENCAO: sanitiza run + volumes"'
	@echo -e "$(RED)  Limpando checkpoints/artefactos + volumes Redis/Timescale/MinIO; recria a stack$(RESET)"
	@echo ""
	@$(MAKE) --no-print-directory docker-sanitize-run-keep-meta
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@$(MAKE) --no-print-directory docker-up

docker-clean:
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/sh/docker-ui.sh; docker_ui_banner "docker-clean · ATENCAO: volumes serao apagados"'
	@echo -e "$(RED)  Removendo containers, redes e volumes (TimescaleDB/MinIO/Redis)$(RESET)"
	@echo ""
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
