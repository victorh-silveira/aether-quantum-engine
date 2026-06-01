# Configurações do projeto
SHELL := /bin/bash
APP_DIR=app
PYTHON=python

# Cores para o terminal (ANSI)
GREEN  := \033[1;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m
CYAN   := \033[1;36m
RESET  := \033[0m

.DEFAULT_GOAL := help

.PHONY: install lint test security run backtest pre-commit clean help helpo

help:
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "$(GREEN)                   AETHER QUANTUM ENGINE - MENU DE AJUDA                $(RESET)"
	@echo -e "$(BLUE)========================================================================$(RESET)"
	@echo -e "Uso: $(CYAN)make <comando>$(RESET)"
	@echo -e ""
	@echo -e "$(YELLOW)Comandos Disponíveis:$(RESET)"
	@echo -e "  $(GREEN)install$(RESET)     - Atualiza o pip e instala as dependências do projeto"
	@echo -e "  $(GREEN)lint$(RESET)        - Roda os linters e verificadores de formatação (Ruff, pylint, etc.)"
	@echo -e "  $(GREEN)test$(RESET)        - Roda os testes unitários com pytest e gera cobertura de código"
	@echo -e "  $(GREEN)security$(RESET)    - Varre o projeto em busca de vulnerabilidades (bandit/pip-audit)"
	@echo -e "  $(GREEN)run$(RESET)         - Inicia a execução principal do motor quântico (run.py)"
	@echo -e "  $(GREEN)backtest$(RESET)    - Executa simulações históricas (use: ARGS=\"...\")"
	@echo -e "  $(GREEN)pre-commit$(RESET)  - Instala e configura os git hooks locais de pre-commit"
	@echo -e "  $(GREEN)clean$(RESET)       - Limpa lixo, caches do Python/Pytest e logs do workspace"
	@echo -e "  $(GREEN)help / helpo$(RESET) - Exibe este menu de ajuda interativo"
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

backtest:
	$(PYTHON) $(APP_DIR)/scripts/backtest/medallion_backtest.py $(ARGS)

pre-commit:
	bash linters/git-hooks/install.sh

clean:
	$(PYTHON) $(APP_DIR)/scripts/operations/clean_workspace.py --stage clean
