SHELL := /bin/bash
.DEFAULT_GOAL := help

APP_DIR := app
DOCKER_DIR := infra/docker
DOCKER_COMPOSE := docker compose -f $(DOCKER_DIR)/docker-compose.yml
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3.13)
PYTHON_CMD := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
DOCKER_LOGS_TAIL ?= 200
DOCKER_LOGS_SERVICES ?= indicator timescaledb prometheus grafana
DOCKER_SERVICE ?=

define docker_service_name
$(strip $(or $(if $(filter ts timescale timescaledb,$(1)),timescaledb),$(if $(filter prom prometheus,$(1)),prometheus),$(if $(filter grafana,$(1)),grafana),$(if $(filter app run indicator,$(1)),indicator),$(1)))
endef

.PHONY: help app-run app-train app-test app-lint app-security app-clean app-install \
	app-pre-commit app-pre-commit-run app-setup-wsl app-rust-lint app-rust-validate \
	app-rust-security app-rust-test app-rust-build docker-up docker-down \
	docker-restart docker-rebuild docker-reset docker-clean docker-ps docker-logs \
	docker-smoke docker-bash docker-hydrate docker-timescale-lifecycle \
	docker-sanitize-run docker-sanitize-run-keep-meta pre-commit timescale-lifecycle \
	test lint train run infra-up infra-down

help:
	@echo 'AETHER SYNTHETIC INDICATOR - MENU DE AJUDA'
	@echo 'Uso: make <comando>'
	@echo 'Python: $(PYTHON) (WSL, Python 3.13)'
	@echo ''
	@echo 'App:'
	@echo '  app-install         Instala dependencias na .venv'
	@echo '  app-run             Executa o indicador local (pare o servico indicator antes)'
	@echo '  app-train           Treina modelos Volatility M5/M15/H1'
	@echo '  app-test            Testes e cobertura de 100%'
	@echo '  app-lint            Ruff lint e formato'
	@echo '  app-security        Ruff security'
	@echo '  app-clean           Remove caches locais de Python'
	@echo '  app-pre-commit      Instala os hooks do repositorio'
	@echo '  app-pre-commit-run  Executa todos os hooks'
	@echo '  app-setup-wsl       Prepara .venv com Python 3.13 no WSL'
	@echo ''
	@echo 'Docker:'
	@echo '  docker-up           Sobe indicador, Timescale, Prometheus e Grafana'
	@echo '  docker-down         Para a stack e preserva volumes'
	@echo '  docker-restart      Reinicia a stack'
	@echo '  docker-rebuild      Reconstroi a imagem do indicador'
	@echo '  docker-reset        Remove volumes do indicador e sobe a stack novamente'
	@echo '  docker-clean        Remove containers, rede e volumes do indicador'
	@echo '  docker-ps           Exibe estado dos containers'
	@echo '  docker-logs         Logs (DOCKER_SERVICE=grafana F=1 DOCKER_LOGS_TAIL=200)'
	@echo '  docker-smoke        Valida endpoints e saude da stack'
	@echo '  docker-bash         Shell no servico (DOCKER_SERVICE=indicator)'
	@echo ''
	@echo 'Compatibilidade: make lint/test/train/run e infra-up/infra-down'

app-install:
	@test -x .venv/bin/python || python3.13 -m venv .venv
	.venv/bin/python -m pip install -r $(APP_DIR)/requirements-dev.txt

app-lint:
	cd $(APP_DIR) && $(PYTHON_CMD) -m ruff check src tests run.py train.py ../run.py ../train.py
	cd $(APP_DIR) && $(PYTHON_CMD) -m ruff format --check src tests run.py train.py ../run.py ../train.py

app-test:
	cd $(APP_DIR) && PYTHONPATH=src $(PYTHON_CMD) -m pytest tests --cov=src --cov-report=term-missing --cov-fail-under=100

app-security:
	$(PYTHON) -m ruff check --select S --ignore S104 app/src app/run.py app/train.py

app-clean:
	find app -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -f app/.coverage .coverage

app-run:
	$(PYTHON) run.py

app-train:
	$(PYTHON) app/train.py

app-pre-commit:
	$(PYTHON) -m pre_commit install -c .pre-commit-config.yaml

app-pre-commit-run:
	$(PYTHON) -m pre_commit run --all-files -c .pre-commit-config.yaml

app-setup-wsl:
	@grep -qi microsoft /proc/version || { echo 'Este target requer WSL.' >&2; exit 1; }
	@$(MAKE) --no-print-directory app-install

app-rust-lint app-rust-validate app-rust-security app-rust-test app-rust-build:
	@echo 'Sem componentes Rust nesta branch do indicador.'

docker-up:
	$(DOCKER_COMPOSE) up -d --build
	@$(MAKE) --no-print-directory docker-smoke

docker-down:
	$(DOCKER_COMPOSE) down

docker-restart:
	$(DOCKER_COMPOSE) restart
	@$(MAKE) --no-print-directory docker-smoke

docker-rebuild:
	$(DOCKER_COMPOSE) build --pull indicator
	$(DOCKER_COMPOSE) up -d --force-recreate indicator
	@$(MAKE) --no-print-directory docker-smoke

docker-reset:
	@echo 'ATENCAO: os volumes Timescale e Grafana do indicador serao removidos.'
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	@$(MAKE) --no-print-directory docker-up

docker-clean:
	@echo 'ATENCAO: os volumes Timescale e Grafana do indicador serao removidos.'
	$(DOCKER_COMPOSE) down --volumes --remove-orphans

docker-ps:
	$(DOCKER_COMPOSE) ps

docker-logs:
	$(DOCKER_COMPOSE) logs --tail=$(DOCKER_LOGS_TAIL) $(if $(F),-f,) $(if $(DOCKER_SERVICE),$(call docker_service_name,$(DOCKER_SERVICE)),$(DOCKER_LOGS_SERVICES))

docker-smoke:
	@bash $(DOCKER_DIR)/sh/docker-smoke.sh

docker-bash:
	$(DOCKER_COMPOSE) exec -it $(call docker_service_name,$(or $(DOCKER_SERVICE),timescaledb)) sh -c 'if [ -x /bin/bash ]; then exec /bin/bash; else exec /bin/sh; fi'

docker-hydrate docker-timescale-lifecycle docker-sanitize-run docker-sanitize-run-keep-meta:
	@echo 'O indicador cria tabelas e coleta o catalogo automaticamente; nao ha estado de contrato para hidratar ou sanitizar.'

pre-commit: app-pre-commit
timescale-lifecycle: docker-timescale-lifecycle
test: app-test
lint: app-lint
train: app-train
run: app-run
infra-up: docker-up
infra-down: docker-down
