# Aether Quantum Engine

[![CI/CD](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/victorh-silveira/aether-quantum-engine?label=release)](https://github.com/victorh-silveira/aether-quantum-engine/releases)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Cobertura do motor: 100% exigidos](https://img.shields.io/badge/cobertura%20motor-100%25%20exigidos-2E7D32)](app/pyproject.toml)
[![Cobertura do indicador: 100% exigidos](https://img.shields.io/badge/cobertura%20indicador-100%25%20exigidos-2E7D32)](indicator/app/pyproject.toml)

[![TCN](https://img.shields.io/badge/modelo-TCN-EE4C2C?logo=pytorch&logoColor=white)](docs/engineering-deep-learning.md)
[![LightGBM](https://img.shields.io/badge/meta-LightGBM-16856B)](app/requirements.txt)
[![Polars](https://img.shields.io/badge/dados-Polars-CD792C)](app/requirements.txt)
[![Docker Compose](https://img.shields.io/badge/infra-Docker%20Compose-2496ED?logo=docker&logoColor=white)](infra/docker/docker-compose.yml)
[![Redis](https://img.shields.io/badge/cache-Redis-DC382D?logo=redis&logoColor=white)](infra/docker/docker-compose.yml)
[![TimescaleDB](https://img.shields.io/badge/series%20temporais-TimescaleDB-FDB515)](infra/docker/docker-compose.yml)
[![MinIO](https://img.shields.io/badge/artefatos-MinIO-C72E49)](infra/docker/docker-compose.yml)

[![Prometheus](https://img.shields.io/badge/metricas-Prometheus-E6522C?logo=prometheus&logoColor=white)](docs/engineering-observability.md)
[![Grafana](https://img.shields.io/badge/dashboards-Grafana-F46800?logo=grafana&logoColor=white)](docs/engineering-observability.md)
[![Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64)](app/requirements-dev.txt)
[![Vulture](https://img.shields.io/badge/codigo%20morto-Vulture%20100-555555)](app/pyproject.toml)
[![Bandit](https://img.shields.io/badge/seguranca-Bandit-555555)](app/scripts/operations/clean_workspace.py)
[![pip-audit](https://img.shields.io/badge/dependencias-pip--audit-555555)](app/scripts/operations/clean_workspace.py)
[![Gitleaks](https://img.shields.io/badge/segredos-Gitleaks-555555)](.github/workflows/ci.yml)
[![Trivy](https://img.shields.io/badge/containers-Trivy-555555)](.github/workflows/ci.yml)

Motor assíncrono de contratos Rise/Fall para índices Volatility da Deriv. O sinal operacional é o [setup fixo de tendência e recuo](docs/strategy-pullback.md) em M5; M15 e H1 são pesquisa. O processo Python 3.13 roda no host; a stack compartilhada reúne Redis, TimescaleDB, indicador público, Prometheus e Grafana. TCN, meta e loss-classifier permanecem disponíveis apenas para pesquisa.

## Fluxo operacional

1. O indicador público consulta velas fechadas e contínuas por índice e período. EMA20/EMA50, recuo até 0,5 ATR14 e confirmação do rompimento determinam CALL ou PUT. Padrão incompleto ou dado atrasado produz `SEM SINAL`.
2. Um setup M5 válido na DEMO pode receber uma cotação e stake base de 1% da banca disponível mesmo com EV estimado negativo. As primeiras 200 compras únicas, liquidadas pelo broker e atribuídas à mesma versão e índice, formam o piloto.
3. A versão passa a `ativo` apenas se o limite inferior unilateral de 95% do retorno médio líquido dessas 200 compras for positivo. Em REAL, a probabilidade vem da posterior Beta(0,5;0,5) do piloto; a proposta final exige EV positivo e margem de 0,01 acima do break-even após haircut.
4. A entrada expira 30 segundos após a abertura da vela seguinte. Somente o sinal M5 abre contrato de 5 minutos; M15/H1 são pesquisa. A stake base em DEMO e REAL é 1% da banca disponível, com recuperação pelo payout líquido cotado limitada a 5% por ordem. Reserva Redis, deduplicação e reconciliação de compras sem ACK permanecem obrigatórias.

O `make app-train` coleta velas e apresenta frequência e acerto pela abertura e pelo fechamento da vela seguinte como **proxy de velas**. Esse diagnóstico não qualifica contratos. `make app-research-ml` executa a pesquisa TCN/meta/loss separadamente. A operação mantém stop win por sessão, sem stop loss ou teto acumulado de perda. Nenhum setup garante vantagem: sem evidência DEMO suficiente ou cotação REAL favorável, não há compra REAL.

Com o motor em execução, Prometheus coleta suas métricas na porta 9100 e as do indicador na porta interna 9101. O Grafana compartilhado consulta Prometheus e TimescaleDB pelos datasources provisionados. A rota entre Docker no WSL e o motor no Windows está descrita em [observabilidade](docs/engineering-observability.md).

## Executar no WSL

Use o ambiente Conda `deriv-api` com Python 3.13.12. Configure `.env` a partir de `.env.example` com as credenciais da Deriv e da infraestrutura.

```bash
make app-install
make app-train
make app-run
```

O launcher de treino em WSL é `app/scripts/wsl/launch-train-wsl.sh`. O motor usa o checkpoint produzido pelo treino separado (`online_training=false`). Para QA, execute `make app-pre-commit-run` no WSL; o pipeline inclui Vulture com confiança 100, lint, segurança, testes e cobertura de 100% em `app/src`.

## Indicador integrado e múltiplos índices

O indicador público em [`indicator/`](indicator/README.md) usa apenas `active_symbols` e `ticks_history`; não possui credenciais de negociação. O Grafana em `localhost:3000` separa setup atual, catálogo completo, piloto DEMO, aptidão REAL, cotações e resultados confirmados. As versões antigas de ML permanecem no histórico, sem autorização para o setup vigente.

```bash
make app-train
make app-run
```

`make docker-up` sobe a stack única. `make app-train` garante a stack e prepara o diagnóstico M5/M15/H1. `make app-run` inicia o motor na conta configurada. Para testar a estratégia, mantenha `trading.mode=demo`; a configuração atual está em DEMO. `make app-lint` e `make app-test` validam motor e indicador. Índices diferentes podem manter contratos simultâneos, com no máximo um contrato aberto por índice. O painel não interpola lacunas e não transforma sinais sintéticos em previsões de Forex.

## Documentação

- [Arquitetura e componentes](docs/arquitetura.md)
- [Treino e checkpoint TCN](docs/engineering-deep-learning.md)
- [Configuração SSOT](docs/engineering-settings-ssot.md)
- [Execução e risco](docs/binary-senior-playbook.md)
- [Infraestrutura Docker](docs/infra-docker.md)
- [Operação e regras para agentes](AGENTS.md)
- [CI/CD e workflows](.github/workflows/WORKFLOWS.md)
