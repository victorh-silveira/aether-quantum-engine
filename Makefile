# Configuracoes do projeto
SHELL := /bin/bash
APP_DIR=app
CONDA_ENV ?= deriv-api
DOCKER_DIR=infra/docker
DOCKER_COMPOSE=docker compose -f $(DOCKER_DIR)/docker-compose.yml --project-directory $(DOCKER_DIR)
DOCKER_SERVICE ?= timescaledb
RESOLVE_PY := $(shell bash linters/git-hooks/bin/resolve_conda_python.sh 2>/dev/null || echo python)
PYTHON := $(RESOLVE_PY)

# Cores para o terminal (ANSI)
GREEN  := \033[1;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
CYAN   := \033[1;36m
RESET  := \033[0m

.DEFAULT_GOAL := help

.PHONY: install lint test security run train pre-commit pre-commit-run setup-wsl clean help helpo \
	docker-up docker-down docker-ps docker-logs docker-bash

help:
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "$(GREEN)                   AETHER QUANTUM ENGINE - MENU DE AJUDA                $(RESET)"
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "Uso: $(CYAN)make <comando>$(RESET)"
	@echo -e ""
	@echo -e "$(YELLOW)Python:$(RESET) Conda $(CONDA_ENV) ($(PYTHON))"
	@echo -e ""
	@echo -e "$(YELLOW)Comandos Disponiveis:$(RESET)"
	@echo -e "  $(GREEN)app-install$(RESET)         - Instala dependencias no Conda $(CONDA_ENV)"
	@echo -e "  $(GREEN)app-lint$(RESET)        	- Roda os linters e verificadores de formatacao (Ruff, pylint, etc.)"
	@echo -e "  $(GREEN)app-test$(RESET)        	- Roda os testes unitarios com pytest e gera cobertura de codigo"
	@echo -e "  $(GREEN)app-security$(RESET)    	- Varre o projeto em busca de vulnerabilidades (bandit/pip-audit)"
	@echo -e "  $(GREEN)app-run$(RESET)         	- Inicia a execucao principal do motor quantico (run.py)"
	@echo -e "  $(GREEN)app-train$(RESET)       	- Treina modelos Deep Learning (train.py)"
	@echo -e "  $(GREEN)app-pre-commit$(RESET)      - Instala e configura os git hooks locais de pre-commit"
	@echo -e "  $(GREEN)app-pre-commit-run$(RESET)  - Roda todos os hooks (equivalente a pre-commit run --all-files)"
	@echo -e "  $(GREEN)app-setup-wsl$(RESET)     	- Configura Git, Conda e hooks no WSL (bash app/scripts/wsl/setup.sh)"
	@echo -e "  $(GREEN)app-clean$(RESET)       	- Limpa lixo, caches do Python/Pytest e logs do workspace"
	@echo -e "  $(GREEN)help / helpo$(RESET) 		- Exibe este menu de ajuda interativo"
	@echo -e ""
	@echo -e "$(YELLOW)Docker:$(RESET)"
	@echo -e "  $(GREEN)docker-up$(RESET)    - Sobe Redis, TimescaleDB e MinIO"
	@echo -e "  $(GREEN)docker-down$(RESET)  - Para e remove containers"
	@echo -e "  $(GREEN)docker-ps$(RESET)    - Status dos containers"
	@echo -e "  $(GREEN)docker-logs$(RESET)  - Logs (DOCKER_SERVICE=redis F=1)"
	@echo -e "  $(GREEN)docker-bash$(RESET)  - Shell no container (DOCKER_SERVICE=timescaledb)"
	@echo -e "$(BLUE)========================================================================$(RESET)"

helpo: help

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(APP_DIR)/requirements.txt -r $(APP_DIR)/requirements-dev.txt

lint:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage lint

test:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage test

security:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage security

run:
	$(PYTHON) run.py

train:
	$(PYTHON) train.py

pre-commit:
	bash linters/git-hooks/install.sh
	chmod +x linters/git-hooks/bin/resolve_conda_python.sh linters/git-hooks/bin/python

pre-commit-run:
	$(PYTHON) -m pre_commit run --all-files -c .pre-commit-config.yaml

setup-wsl:
	bash app/scripts/wsl/setup.sh

clean:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage clean

docker-up:
	@test -f $(DOCKER_DIR)/.env || cp $(DOCKER_DIR)/.env.example $(DOCKER_DIR)/.env
	$(DOCKER_COMPOSE) up -d
	@$(DOCKER_COMPOSE) ps

docker-down:
	$(DOCKER_COMPOSE) down

docker-ps:
	$(DOCKER_COMPOSE) ps

docker-logs:
	$(DOCKER_COMPOSE) logs $(if $(F),-f,) $(if $(DOCKER_SERVICE),$(DOCKER_SERVICE),)

docker-bash:
	$(DOCKER_COMPOSE) exec -it $(DOCKER_SERVICE) sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'
