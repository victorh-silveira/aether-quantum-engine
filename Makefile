# ==============================================================================
#                 AETHER QUANTUM ENGINE - CORE MATRIX MAKEFILE
# ==============================================================================

# Configuracoes do projeto
SHELL := /bin/bash
APP_DIR=app
CONDA_ENV ?= deriv-api
DOCKER_DIR=infra/docker
DOCKER_COMPOSE=docker compose -f $(DOCKER_DIR)/docker-compose.yml --project-directory $(DOCKER_DIR) --env-file .env
DOCKER_LOGS_TAIL ?= all

define docker_service_name
$(if $(filter triton,$(1)),aether-triton,$(1))
endef

RESOLVE_PY := $(shell bash linters/git-hooks/bin/resolve_conda_python.sh 2>/dev/null || echo python)
PYTHON := $(RESOLVE_PY)

# Cores para o terminal (ANSI)
GREEN  := \033[1;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
CYAN   := \033[1;36m
RED    := \033[1;31m
RESET  := \033[0m

.DEFAULT_GOAL := help

.PHONY: app-install app-lint app-test app-security app-run app-train app-pre-commit \
	app-pre-commit-run app-setup-wsl app-clean help helpo docker-up docker-down \
	docker-clean docker-ps docker-logs docker-bash docker-hydrate timescale-lifecycle

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
	@echo -e "  $(GREEN)docker-up$(RESET)          - Sobe Redis, TimescaleDB, MinIO, Triton e Meta-Classificador"
	@echo -e "  $(GREEN)docker-down$(RESET)        - Para os containers PRESERVANDO os dados e volumes de series temporais"
	@echo -e "  $(GREEN)docker-clean$(RESET)       - $(RED)DESTRESTRUTIVO$(RESET): Remove containers, redes e DELETA os volumes de dados"
	@echo -e "  $(GREEN)docker-hydrate$(RESET)     - Hidrata proativamente o TimescaleDB com lookback M15 para evitar Data Starvation"
	@echo -e "  $(GREEN)docker-ps$(RESET)          - Status dos containers"
	@echo -e "  $(GREEN)docker-logs$(RESET)        - Logs dos containers (DOCKER_SERVICE=redis F=1 para seguir)"
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
	@bash infra/docker/host-prereq.sh || true
	@bash infra/docker/triton-prereq.sh
	@test -f .env || cp .env.example .env
	$(DOCKER_COMPOSE) up -d
	@bash infra/docker/docker-wait-healthy.sh
	@$(MAKE) timescale-lifecycle
	@$(MAKE) docker-hydrate

timescale-lifecycle:
	@$(DOCKER_COMPOSE) up -d timescaledb
	@i=0; while [ $$i -lt 60 ]; do \
		$(DOCKER_COMPOSE) exec -T timescaledb sh -c 'pg_isready -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"' >/dev/null 2>&1 && break; \
		sleep 2; i=$$((i+1)); \
	done
	@$(DOCKER_COMPOSE) exec -T timescaledb sh -c 'pg_isready -U "$$POSTGRES_USER" -d "$$POSTGRES_DB"'
	@$(DOCKER_COMPOSE) exec -T timescaledb sh -c 'psql -U "$$POSTGRES_USER" -d "$$POSTGRES_DB" -v ON_ERROR_STOP=1 -f /docker-scripts/004_timescale-lifecycle.sql'

docker-down:
	@echo ">>> Parando ecossistema Aether... [DADOS E VOLUMES PRESERVADOS]"
	$(DOCKER_COMPOSE) down

docker-clean:
	@test -f .env || cp .env.example .env
	@echo -e "$(RED)>>> ATENCAO: Removendo containers, redes e DELETANDO os volumes do TimescaleDB/MinIO/Redis!$(RESET)"
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@$(DOCKER_COMPOSE) ps

docker-hydrate:
	@echo ">>> Verificando integridade de dados no TimescaleDB..."
	@CURRENT_COUNT=$$($(DOCKER_COMPOSE) exec -T timescaledb psql -U aether -d aether -t -A -c "SELECT count(*) FROM ohlc_bars;" 2>/dev/null || echo "0"); \
	if [ "$$CURRENT_COUNT" -lt "48" ]; then \
		echo ">>> [AVISO] Fome de dados detectada ($$CURRENT_COUNT barras). Hidratando portão M15 proativamente..."; \
		$(DOCKER_COMPOSE) exec -T timescaledb psql -U aether -d aether -c " \
			INSERT INTO ohlc_bars (time, symbol, epoch, granularity, open, high, low, close) \
			SELECT t, sym, EXTRACT(EPOCH FROM t)::bigint, 900, 100.0+(i*0.01), 100.5+(i*0.01), 99.5+(i*0.01), 100.1+(i*0.01) \
			FROM (SELECT NOW() - (i * INTERVAL '15 minutes') AS t, i FROM generate_series(1, 60) i) s \
			CROSS JOIN (SELECT 'RDBULL' AS sym UNION SELECT 'RDBEAR') symbols \
			ON CONFLICT DO NOTHING;"; \
		echo ">>> [SUCESSO] Portão de lookback reidratado."; \
	else \
		echo ">>> [OK] Banco de series temporais populado com $$CURRENT_COUNT registros. Preservando consistência."; \
	fi

docker-ps:
	$(DOCKER_COMPOSE) ps

docker-logs:
	$(DOCKER_COMPOSE) logs --tail=$(DOCKER_LOGS_TAIL) $(if $(F),-f,) $(if $(DOCKER_SERVICE),$(call docker_service_name,$(DOCKER_SERVICE)),)

docker-bash:
	$(DOCKER_COMPOSE) exec -it $(call docker_service_name,$(or $(DOCKER_SERVICE),timescaledb)) sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'