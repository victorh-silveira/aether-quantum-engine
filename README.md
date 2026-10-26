# Aether Quantum Engine

[![CI/CD](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/victorh-silveira/aether-quantum-engine?label=release)](https://github.com/victorh-silveira/aether-quantum-engine/releases)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Cobertura do motor: 100% exigidos](https://img.shields.io/badge/cobertura%20motor-100%25%20exigidos-2E7D32)](app/pyproject.toml)
[![Cobertura do indicador: 100% exigidos](https://img.shields.io/badge/cobertura%20indicador-100%25%20exigidos-2E7D32)](indicator/app/pyproject.toml)

[![TCN](https://img.shields.io/badge/modelo-TCN-EE4C2C?logo=pytorch&logoColor=white)](docs/engineering-deep-learning.md)
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

Motor assíncrono de contratos Rise/Fall para índices OTC da Deriv. O caminho operacional usa o [setup institucional M15](docs/strategy-pullback.md) com tendência EMA20/EMA50, recuo, referência de preço da sessão, FVG mitigado e killzones BRT. O indicador público e o motor compartilham TimescaleDB, Prometheus e Grafana. A referência da sessão usa OHLC sem volume; não é VWAP por volume.

## Fluxo operacional

1. O indicador calcula sinais por ativo e período apenas com velas fechadas da Deriv. Somente M15 da versão `senior_institutional` configurada pode gerar contrato de 15 minutos. H1 e D1 permanecem pesquisa.
2. A confirmação precisa estar na killzone BRT e satisfazer todos os pilares. Sem padrão, com vela atrasada ou fora dos primeiros 60 segundos da vela seguinte, não há compra. O motor conserva o lado CALL/PUT publicado; não usa fallback do stream M5.
3. Em DEMO, até 200 compras por ativo, período e versão compõem o piloto. Apenas liquidações únicas confirmadas pelo broker contam. O treino fornece diagnóstico histórico e não promove contratos.
4. REAL exige 200 liquidações DEMO, limite inferior unilateral de 95% do retorno médio líquido positivo e EV positivo com margem de 0,01 sobre o break-even na cotação final. Stake base de 1% da banca disponível; recuperação pelo payout cotado limitada a 5% por ordem.

O `make app-train` diagnostica M15, H1 e D1 do catálogo público sem misturar preços Yahoo com as velas da Deriv e sem alterar a configuração operacional. A âncora configurada é **`OTC_SSMI`**. O ranking do proxy não comprova vantagem de contrato. A operação mantém stop win por sessão, sem stop loss ou teto acumulado de perda.

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

`make docker-up` sobe a stack única. `make app-train` garante a stack e prepara o diagnóstico M15/H1/D1. `make app-run` inicia o motor na conta configurada. Para testar a estratégia, mantenha `trading.mode=demo`; a configuração atual está em DEMO. `make app-lint` e `make app-test` validam motor e indicador. Índices diferentes podem manter contratos simultâneos, com no máximo um contrato aberto por índice. O painel não interpola lacunas e não transforma sinais sintéticos em previsões de Forex.

## Documentação

- [Arquitetura e componentes](docs/arquitetura.md)
- [Treino e checkpoint TCN](docs/engineering-deep-learning.md)
- [Configuração SSOT](docs/engineering-settings-ssot.md)
- [Execução e risco](docs/binary-senior-playbook.md)
- [Infraestrutura Docker](docs/infra-docker.md)
- [Operação e regras para agentes](AGENTS.md)
- [CI/CD e workflows](.github/workflows/WORKFLOWS.md)
