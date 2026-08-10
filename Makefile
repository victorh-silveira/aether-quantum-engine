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
$(strip $(or \
	$(if $(filter redis aether-redis,$(1)),redis),\
	$(if $(filter ts timescale timescaledb aether-timescaledb,$(1)),timescaledb),\
	$(if $(filter minio aether-minio,$(1)),minio),\
	$(if $(filter triton aether-triton,$(1)),aether-triton),\
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

.PHONY: app-install app-lint app-test app-security app-run app-train app-pre-commit \
	app-pre-commit-run app-setup-wsl app-clean help helpo docker-up docker-up-core \
	docker-up-cpu docker-down docker-clean docker-restart docker-reset docker-ps docker-logs \
	docker-bash docker-hydrate docker-rebuild docker-smoke timescale-lifecycle sanitize-run \
	sanitize-run-docker


help:
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "$(GREEN)                   AETHER QUANTUM ENGINE - MENU DE AJUDA                $(RESET)"
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "Uso: $(CYAN)make <comando>$(RESET)"
	@echo -e ""
	@echo -e "$(YELLOW)Python:$(RESET) Conda $(CONDA_ENV) ($(PYTHON))"
	@echo -e ""
	@echo -e "$(YELLOW)App:$(RESET)"
	@echo -e "  $(GREEN)app-run$(RESET)            - Sobe o motor (run.py)"
	@echo -e "  $(GREEN)app-train$(RESET)          - Treina DL (train.py)"
	@echo -e "  $(GREEN)app-test$(RESET)           - Testes + cobertura 100%"
	@echo -e "  $(GREEN)app-lint$(RESET)           - Lint / format"
	@echo -e "  $(GREEN)app-clean$(RESET)          - Limpa caches/logs locais"
	@echo -e "  $(GREEN)app-install$(RESET)        - Pip no Conda $(CONDA_ENV)"
	@echo -e ""
	@echo -e "$(YELLOW)Docker:$(RESET)"
	@echo -e "  $(GREEN)docker-up$(RESET)          - Stack completa GPU"
	@echo -e "  $(GREEN)docker-up-cpu$(RESET)      - Stack Triton CPU"
	@echo -e "  $(GREEN)docker-up-core$(RESET)     - So Redis/Timescale/MinIO"
	@echo -e "  $(GREEN)docker-rebuild$(RESET)     - Limpa loss-models, bootstrap cold-start, rebuild meta/loss e sobe"
	@echo -e "  $(GREEN)docker-reset$(RESET)       - $(RED)DESTRUTIVO$(RESET): sanitiza run + loss-models + volumes, bootstrap e sobe stack"
	@echo -e "  $(GREEN)sanitize-run$(RESET)       - $(RED)DESTRUTIVO$(RESET): limpa checkpoints DL/meta/loss/triton e data/ (exceto deriv)"
	@echo -e "  $(GREEN)docker-down$(RESET)        - Para containers (preserva dados)"
	@echo -e "  $(GREEN)docker-restart$(RESET)     - Restart da stack"
	@echo -e "  $(GREEN)docker-ps$(RESET)          - Status"
	@echo -e "  $(GREEN)docker-logs$(RESET)        - Logs (DOCKER_SERVICE=... F=1)"
	@echo -e "  $(GREEN)docker-smoke$(RESET)       - Smoke checks"
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

sanitize-run:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "sanitize-run · limpa checkpoints e artefactos da run anterior"'
	$(PYTHON) $(APP_DIR)/scripts/operations/sanitize_fresh_run.py

sanitize-run-docker:
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "sanitize-run-docker · limpa run (mantem meta_lgbm.pkl ate train)"'
	$(PYTHON) $(APP_DIR)/scripts/operations/sanitize_fresh_run.py --keep-meta-bundle

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
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-rebuild · sanitiza run + limpa loss-models + bootstrap + rebuild meta/loss + up"'
	@test -f .env || cp .env.example .env
	@$(MAKE) --no-print-directory sanitize-run-docker
	@bash infra/docker/loss-clf-reset.sh clear
	@cd $(APP_DIR) && LOKY_MAX_CPU_COUNT=$${LOKY_MAX_CPU_COUNT:-4} $(PYTHON) -m scripts.operations.train_loss_classifier
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
	@bash -c 'source infra/docker/docker-ui.sh; docker_ui_banner "docker-reset · ATENCAO: sanitiza run + loss-models + volumes"'
	@echo -e "$(RED)  Limpando checkpoints/artefactos + loss-models + volumes Redis/Timescale/MinIO; recria a stack$(RESET)"
	@echo ""
	@$(MAKE) --no-print-directory sanitize-run-docker
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
	$(DOCKER_COMPOSE) logs --tail=$(DOCKER_LOGS_TAIL) $(if $(F),-f,) $(if $(DOCKER_SERVICE),$(call docker_service_name,$(DOCKER_SERVICE)),)

docker-bash:
	$(DOCKER_COMPOSE) exec -it $(call docker_service_name,$(or $(DOCKER_SERVICE),timescaledb)) sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'
