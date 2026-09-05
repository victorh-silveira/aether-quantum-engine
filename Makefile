# ==============================================================================
#                 AETHER QUANTUM ENGINE - CORE MATRIX MAKEFILE
# ==============================================================================

SHELL := /bin/bash
APP_DIR=app
CONDA_ENV ?= deriv-api
DOCKER_DIR=infra/docker
DOCKER_PROFILES ?= core,ml
export COMPOSE_PROFILES := $(DOCKER_PROFILES)
DOCKER_COMPOSE=docker compose -f $(DOCKER_DIR)/docker-compose.yml --project-directory $(DOCKER_DIR) --env-file .env
DOCKER_LOGS_TAIL ?= 200
DOCKER_LOGS_SERVICES ?= redis timescaledb minio aether-meta-classifier aether-loss-classifier

define docker_service_name
$(strip $(or \
	$(if $(filter redis aether-redis,$(1)),redis),\
	$(if $(filter ts timescale timescaledb aether-timescaledb,$(1)),timescaledb),\
	$(if $(filter minio aether-minio,$(1)),minio),\
	$(if $(filter meta meta-classifier aether-meta-classifier,$(1)),aether-meta-classifier),\
	$(if $(filter loss loss-classifier aether-loss-classifier,$(1)),aether-loss-classifier),\
	$(1)))
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

.PHONY: app-run app-train app-test app-lint app-security app-clean app-install \
	app-pre-commit app-pre-commit-run app-setup-wsl help docker-up docker-down \
	docker-restart docker-rebuild docker-reset docker-clean docker-ps docker-logs \
	docker-smoke docker-bash docker-hydrate docker-timescale-lifecycle


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
	@echo -e "  $(GREEN)app-train$(RESET)          - Pipeline completo de treino DL 5m (TCN + meta-classificador)"
	@echo -e "  $(GREEN)app-test$(RESET)           - Testes automatizados + cobertura 100%"
	@echo -e "  $(GREEN)app-lint$(RESET)           - Lint / format (Ruff + Interrogate + Vulture)"
	@echo -e "  $(GREEN)app-security$(RESET)       - Auditoria de seguranca de codigo (Bandit)"
	@echo -e "  $(GREEN)app-clean$(RESET)          - Limpa caches, artefactos e logs locais"
	@echo -e "  $(GREEN)app-install$(RESET)        - Instala dependencias no Conda $(CONDA_ENV)"
	@echo -e "  $(GREEN)app-pre-commit$(RESET)     - Instala e valida githooks pre-commit"
	@echo -e "  $(GREEN)app-pre-commit-run$(RESET) - Executa pre-commit em todos os arquivos"
	@echo -e ""
	@echo -e "$(YELLOW)Docker:$(RESET)"
	@echo -e "  $(GREEN)docker-up$(RESET)          - Sobe a stack completa (core + ml)"
	@echo -e "  $(GREEN)docker-rebuild$(RESET)     - Rebuilda meta/loss e recarrega pkls (preserva TCN e meta_lgbm)"
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

app-lint:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage lint

app-test:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage test

app-security:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage security

app-clean:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage clean

app-run:
	$(PYTHON) run.py

app-train:
	$(PYTHON) $(APP_DIR)/scripts/operations/run_launch_train_tf_pipeline.py

app-pre-commit:
	bash linters/git-hooks/install.sh
	chmod +x linters/git-hooks/bin/resolve_conda_python.sh linters/git-hooks/bin/python

app-pre-commit-run:
	$(PYTHON) -m pre_commit run --all-files -c .pre-commit-config.yaml

app-setup-wsl:
	bash app/scripts/wsl/setup.sh

# ------------------------------------------------------------------------------
# Docker Targets
# ------------------------------------------------------------------------------

docker-sanitize-run:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-sanitize-run · limpa checkpoints e artefactos da run anterior"'
	$(PYTHON) $(APP_DIR)/scripts/operations/sanitize_fresh_run.py

docker-sanitize-run-keep-meta:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-sanitize-run-keep-meta · limpa TCN/loss/estado (mantem meta_lgbm.pkl)"'
	$(PYTHON) $(APP_DIR)/scripts/operations/sanitize_fresh_run.py --keep-meta-bundle

docker-up:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-up · Aether stack (profiles: $(DOCKER_PROFILES))"'
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 1 5 "Host prerequisites"'
	@bash infra/docker/host-prereq.sh
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 2 5 "Compose up"'
	$(DOCKER_COMPOSE) up -d
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 3 5 "Healthchecks"'
	@bash infra/docker/docker-wait-healthy.sh
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 4 5 "Timescale lifecycle + hydrate"'
	@bash infra/docker/timescale-lifecycle.sh
	@bash infra/docker/docker-hydrate.sh
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_step 5 5 "Smoke checks"'
	@bash infra/docker/docker-smoke.sh

docker-rebuild:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-rebuild · rebuild meta/loss (preserva data/dl e meta_lgbm.pkl)"'
	@test -f .env || cp .env.example .env
	@if [ ! -f $(DOCKER_DIR)/loss-models/loss_bootstrap_synth.pkl ]; then \
		cd $(APP_DIR) && LOKY_MAX_CPU_COUNT=$${LOKY_MAX_CPU_COUNT:-4} $(PYTHON) -m scripts.operations.train_loss_classifier; \
	fi
	$(DOCKER_COMPOSE) build --pull aether-meta-classifier aether-loss-classifier
	$(DOCKER_COMPOSE) up -d --force-recreate aether-meta-classifier aether-loss-classifier
	$(DOCKER_COMPOSE) up -d
	@bash infra/docker/docker-wait-healthy.sh
	@bash infra/docker/docker-smoke.sh

docker-timescale-lifecycle:
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
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-reset · ATENCAO: sanitiza run + loss-models + volumes"'
	@echo -e "$(RED)  Limpando checkpoints/artefactos + loss-models + volumes Redis/Timescale/MinIO; recria a stack$(RESET)"
	@echo ""
	@$(MAKE) --no-print-directory docker-sanitize-run-keep-meta
	@bash infra/docker/loss-clf-reset.sh clear
	@cd $(APP_DIR) && LOKY_MAX_CPU_COUNT=$${LOKY_MAX_CPU_COUNT:-4} $(PYTHON) -m scripts.operations.train_loss_classifier
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@$(MAKE) --no-print-directory docker-up

docker-clean:
	@test -f .env || cp .env.example .env
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-clean · ATENCAO: volumes serao apagados"'
	@echo -e "$(RED)  Removendo containers, redes e volumes (TimescaleDB/MinIO/Redis)$(RESET)"
	@echo ""
	@bash infra/docker/loss-clf-reset.sh clear
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
	$(DOCKER_COMPOSE) logs --tail=$(DOCKER_LOGS_TAIL) $(if $(F),-f,) $(if $(DOCKER_SERVICE),$(call docker_service_name,$(DOCKER_SERVICE)),$(DOCKER_LOGS_SERVICES))

docker-bash:
	$(DOCKER_COMPOSE) exec -it $(call docker_service_name,$(or $(DOCKER_SERVICE),timescaledb)) sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'

# ------------------------------------------------------------------------------
# Aliases de Compatibilidade
# ------------------------------------------------------------------------------
pre-commit: app-pre-commit
timescale-lifecycle: docker-timescale-lifecycle
