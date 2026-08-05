# ==============================================================================
#                 AETHER QUANTUM ENGINE - CORE MATRIX MAKEFILE
# ==============================================================================

SHELL := /bin/bash
APP_DIR=app
CONDA_ENV ?= deriv-api
DOCKER_DIR=infra/docker
DOCKER_PROFILES ?= core,gpu,ml
DOCKER_GPU ?= 1
export COMPOSE_PROFILES := $(DOCKER_PROFILES)
export DOCKER_GPU
DOCKER_COMPOSE_BASE=docker compose -f $(DOCKER_DIR)/docker-compose.yml
ifeq ($(DOCKER_GPU),1)
DOCKER_COMPOSE=$(DOCKER_COMPOSE_BASE) -f $(DOCKER_DIR)/docker-compose.gpu.yml --project-directory $(DOCKER_DIR) --env-file .env
else
DOCKER_COMPOSE=$(DOCKER_COMPOSE_BASE) --project-directory $(DOCKER_DIR) --env-file .env
endif
DOCKER_LOGS_TAIL ?= all

define docker_service_name
$(if $(filter triton,$(1)),aether-triton,$(if $(filter meta,$(1)),aether-meta-classifier,$(1)))
endef

RESOLVE_PY := $(shell bash linters/git-hooks/bin/resolve_conda_python.sh 2>/dev/null || echo python)
PYTHON := $(RESOLVE_PY)

GREEN  := \033[1;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
CYAN   := \033[1;36m
RED    := \033[1;31m
RESET  := \033[0m

.DEFAULT_GOAL := help

.PHONY: app-install app-lint app-test app-security app-run app-train app-pre-commit \
	app-pre-commit-run app-setup-wsl app-clean help helpo docker-up docker-up-core \
	docker-up-cpu docker-down docker-clean docker-restart docker-reset docker-ps docker-logs \
	docker-bash docker-hydrate docker-rebuild docker-smoke timescale-lifecycle

help:
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "$(GREEN)                   AETHER QUANTUM ENGINE - MENU DE AJUDA                $(RESET)"
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "Uso: $(CYAN)make <comando>$(RESET)"
	@echo -e ""
	@echo -e "$(YELLOW)Python:$(RESET) Conda $(CONDA_ENV) ($(PYTHON))"
	@echo -e ""
	@echo -e "$(YELLOW)Comandos Disponiveis (Aplicação):$(RESET)"
	@echo -e "  $(GREEN)app-install$(RESET)        - Instala dependencias no Conda $(CONDA_ENV)"
	@echo -e "  $(GREEN)app-lint$(RESET)           - Roda os linters e verificadores de formatacao (Ruff, etc.)"
	@echo -e "  $(GREEN)app-test$(RESET)           - Roda os testes unitarios com pytest-xdist e cobertura 100%"
	@echo -e "  $(GREEN)app-security$(RESET)       - Varre o projeto em busca de vulnerabilidades (bandit/pip-audit)"
	@echo -e "  $(GREEN)app-run$(RESET)            - Inicia a execucao principal do motor quantico (run.py)"
	@echo -e "  $(GREEN)app-train$(RESET)          - Treina modelos Deep Learning (train.py)"
	@echo -e "  $(GREEN)app-pre-commit$(RESET)     - Instala e configura os git hooks locais de pre-commit"
	@echo -e "  $(GREEN)app-pre-commit-run$(RESET) - Roda todos os hooks (pre-commit run --all-files)"
	@echo -e "  $(GREEN)app-setup-wsl$(RESET)      - Configura Git, Conda e hooks no WSL (setup.sh)"
	@echo -e "  $(GREEN)app-clean$(RESET)          - Limpa caches, logs e dados locais (preserva volumes Docker)"
	@echo -e "  $(GREEN)help / helpo$(RESET)       - Exibe este menu de ajuda interativo"
	@echo -e ""
	@echo -e "$(YELLOW)Comandos Disponiveis (Docker/Infra):$(RESET)"
	@echo -e "  $(GREEN)docker-up$(RESET)          - Stack completa GPU (profiles: $(DOCKER_PROFILES), DOCKER_GPU=$(DOCKER_GPU))"
	@echo -e "  $(GREEN)docker-up-core$(RESET)     - Sobe so Redis, TimescaleDB e MinIO (profile core)"
	@echo -e "  $(GREEN)docker-up-cpu$(RESET)      - Stack com Triton CPU (core,cpu,ml; sem overlay NVIDIA)"
	@echo -e "  $(GREEN)docker-rebuild$(RESET)     - Rebuild meta+loss classifiers + up com profiles ativos"
	@echo -e "  $(GREEN)docker-smoke$(RESET)       - Valida endpoints da stack (Redis/TS/MinIO/Triton/Meta)"
	@echo -e "  $(GREEN)docker-down$(RESET)        - Para os containers PRESERVANDO os dados e volumes"
	@echo -e "  $(GREEN)docker-restart$(RESET)     - Reinicia os containers da stack (volumes preservados)"
	@echo -e "  $(GREEN)docker-reset$(RESET)       - $(RED)DESTRUTIVO$(RESET): Apaga volumes/dados e sobe stack limpa"
	@echo -e "  $(GREEN)docker-clean$(RESET)       - $(RED)DESTRUTIVO$(RESET): Remove containers, redes e DELETA volumes"
	@echo -e "  $(GREEN)docker-hydrate$(RESET)     - Hidrata TimescaleDB macro 600s / micro 120s (R_10)"
	@echo -e "  $(GREEN)docker-ps$(RESET)          - Status dos containers"
	@echo -e "  $(GREEN)docker-logs$(RESET)        - Logs (DOCKER_SERVICE=redis F=1 para seguir)"
	@echo -e "  $(GREEN)docker-bash$(RESET)        - Shell interativo (DOCKER_SERVICE=triton|timescaledb)"
	@echo -e "  $(GREEN)timescale-lifecycle$(RESET) - Aplica compressao/retencao Timescale (idempotente)"
	@echo -e "$(BLUE)========================================================================$(RESET)"

helpo: help

app-install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(APP_DIR)/requirements.txt -r $(APP_DIR)/requirements-dev.txt

app-lint:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage lint

app-test:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage test

app-security:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage security

app-run:
	$(PYTHON) run.py

app-train:
	$(PYTHON) train.py

app-pre-commit:
	bash linters/git-hooks/install.sh
	chmod +x linters/git-hooks/bin/resolve_conda_python.sh linters/git-hooks/bin/python

app-pre-commit-run:
	$(PYTHON) -m pre_commit run --all-files -c .pre-commit-config.yaml

app-setup-wsl:
	bash app/scripts/wsl/setup.sh

app-clean:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage clean

docker-up:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-up · Aether stack (profiles: $(DOCKER_PROFILES))"'
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 1 6 "Host prerequisites"'
	@bash infra/docker/host-prereq.sh
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 2 6 "Triton model layout"'
	@bash infra/docker/triton-prereq.sh
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 3 6 "Compose up"'
	$(DOCKER_COMPOSE) up -d
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 4 6 "Healthchecks"'
	@bash infra/docker/docker-wait-healthy.sh
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 5 6 "Timescale lifecycle + hydrate"'
	@bash infra/docker/timescale-lifecycle.sh
	@bash infra/docker/docker-hydrate.sh
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 6 6 "Smoke checks"'
	@bash infra/docker/docker-smoke.sh

docker-up-core:
	@test -f .env || cp .env.example .env
	@docker stop aether-triton aether-meta-classifier >/dev/null 2>&1 || true
	@$(MAKE) --no-print-directory docker-up DOCKER_PROFILES=core DOCKER_GPU=0

docker-up-cpu:
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-up-cpu · Triton sem NVIDIA (mutuamente exclusivo com GPU)"'
	@$(MAKE) --no-print-directory docker-up DOCKER_PROFILES=core,cpu,ml DOCKER_GPU=0

docker-rebuild:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-rebuild · meta+loss classifiers + stack"'
	@test -f .env || cp .env.example .env
	@bash infra/docker/triton-prereq.sh
	$(DOCKER_COMPOSE) build --pull aether-meta-classifier aether-loss-classifier
	$(DOCKER_COMPOSE) up -d
	@bash infra/docker/docker-wait-healthy.sh
	@bash infra/docker/docker-smoke.sh

timescale-lifecycle:
	@bash infra/docker/timescale-lifecycle.sh

docker-down:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-down · parando stack (volumes preservados)"'
	$(DOCKER_COMPOSE) down
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_nl'

docker-restart:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-restart · reiniciando containers (volumes preservados)"'
	@test -f .env || cp .env.example .env
	$(DOCKER_COMPOSE) restart
	@bash infra/docker/docker-wait-healthy.sh
	@$(DOCKER_COMPOSE) ps

docker-reset:
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-reset · ATENCAO: dados e volumes serao apagados"'
	@echo -e "$(RED)  Limpando remanescentes de runs (Redis/TimescaleDB/MinIO) e recriando a stack$(RESET)"
	@echo ""
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@$(MAKE) --no-print-directory docker-up

docker-clean:
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-clean · ATENCAO: volumes serao apagados"'
	@echo -e "$(RED)  Removendo containers, redes e volumes (TimescaleDB/MinIO/Redis)$(RESET)"
	@echo ""
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@echo ""
	@$(DOCKER_COMPOSE) ps

docker-hydrate:
	@bash infra/docker/docker-hydrate.sh

docker-smoke:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-smoke"'
	@bash infra/docker/docker-smoke.sh

docker-ps:
	$(DOCKER_COMPOSE) ps

docker-logs:
	$(DOCKER_COMPOSE) logs --tail=$(DOCKER_LOGS_TAIL) $(if $(F),-f,) $(if $(DOCKER_SERVICE),$(call docker_service_name,$(DOCKER_SERVICE)),)

docker-bash:
	$(DOCKER_COMPOSE) exec -it $(call docker_service_name,$(or $(DOCKER_SERVICE),timescaledb)) sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'
