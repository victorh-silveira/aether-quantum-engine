SHELL := /bin/bash
.DEFAULT_GOAL := help

APP_DIR := app
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
	@echo '  app-train           Delega o treino completo (indicador + TCN/meta/loss) a raiz'
	@echo '  app-test            Testes e cobertura de 100%'
	@echo '  app-lint            Ruff lint e formato'
	@echo '  app-security        Ruff security'
	@echo '  app-clean           Remove caches locais de Python'
	@echo '  app-pre-commit      Instala os hooks do repositorio'
	@echo '  app-pre-commit-run  Executa todos os hooks'
	@echo '  app-setup-wsl       Prepara .venv com Python 3.13 no WSL'
	@echo ''
	@echo 'Docker:'
	@echo '  docker-up           Sobe a stack unificada na raiz do repositorio'
	@echo '  docker-down         Para a stack unificada e preserva volumes'
	@echo '  docker-restart      Reinicia a stack unificada'
	@echo '  docker-rebuild      Reconstroi os servicos da stack unificada'
	@echo '  docker-reset        Remove volumes da stack unificada e sobe novamente'
	@echo '  docker-clean        Remove containers, rede e volumes da stack unificada'
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
	$(MAKE) -C .. app-train

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
	$(MAKE) -C .. docker-up

docker-down:
	$(MAKE) -C .. docker-down

docker-restart:
	$(MAKE) -C .. docker-restart

docker-rebuild:
	$(MAKE) -C .. docker-rebuild

docker-reset:
	$(MAKE) -C .. docker-reset

docker-clean:
	$(MAKE) -C .. docker-clean

docker-ps:
	$(MAKE) -C .. docker-ps

docker-logs:
	$(MAKE) -C .. docker-logs DOCKER_LOGS_TAIL=$(DOCKER_LOGS_TAIL) DOCKER_SERVICE=$(DOCKER_SERVICE) F=$(F)

docker-smoke:
	$(MAKE) -C .. docker-smoke

docker-bash:
	$(MAKE) -C .. docker-bash DOCKER_SERVICE=$(call docker_service_name,$(or $(DOCKER_SERVICE),timescaledb))

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
