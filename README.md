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

Motor quantitativo assíncrono para contratos Rise/Fall de 5 minutos nos índices Volatility da Deriv. A `main` integra sinais M5 aprovados do indicador ao pipeline TCN, meta e loss-classifier, com uma ordem por ativo elegível no mesmo ciclo. O processo Python 3.13 roda no host com `asyncio`; Redis, TimescaleDB, MinIO, meta LightGBM e loss-classifier rodam como serviços Docker. A arquitetura separa domínio, aplicação, portas, adaptadores e apresentação. O estado tabular usa Polars.

A configuração operacional está em [`config/settings.json`](config/settings.json). O universo e a cadência atuais são M5 de 300 segundos, contexto D1 de 86.400 segundos, 25.000 barras M5 no treino, lookback TCN de 32 barras, horizonte de uma barra e label `spot_forward`. O contrato executado dura 5 minutos. O label por fechamento é uma aproximação do desfecho do contrato e não representa o spot executado.

## Fluxo operacional

1. O WebSocket Deriv recebe ticks e velas; o ciclo abre a cada fronteira M5. O indicador público observa os Volatility em serviço separado e publica sinais por ativo e período.
2. O checkpoint TCN local fornece a probabilidade direcional. CALL é escolhido quando a probabilidade calibrada é pelo menos 0,5; caso contrário, PUT.
3. O loss-classifier pode inverter o lado após aprendizado com contratos reais. O meta LightGBM fornece contexto adicional; filtros técnicos, de mercado e de cotação podem impedir a compra.
4. Kelly fracionário e recuperação calculam a stake. Sem checkpoint técnico compatível, nenhuma ordem é enviada. Com checkpoint válido, o teto inicial é **1% da banca por ordem**, inclusive em `cover_l0` e em DEMO ou REAL.
5. Antes da compra, o sinal M5 **APROVADO**, fresco e da mesma vela precisa concordar com a TCN e com a direção final do motor. Uma inversão de lado pelo meta, loss-classifier ou filtros não satisfaz o consenso. A proposta é revalidada antes da compra. O resultado do contrato é reconciliado com a Deriv e persistido junto com as versões da TCN e do indicador.

O treino salva o checkpoint TCN local após verificar integridade e geometria. Não há qualificação estatística de deploy da TCN nem avaliação diagnóstica de liquidação por closes M5. A validação interna usada para ajuste e calibração não autoriza aumentar o teto inicial. A captura Timescale de ticks e contratos confirmados é independente da decisão de compra.

A política operacional mantém stop win por sessão e **não usa stop loss nem limite acumulado de perda**. O stop win configurado é 4,31% para a banca aplicável; o payout de referência é 0,85. O caminho de recuperação preserva seus limites próprios, sempre subordinados ao teto inicial de 1% quando o checkpoint local é usado.

Com o motor em execução, Prometheus coleta `/metrics` na porta 9100 e o Grafana consulta o Prometheus pelo datasource provisionado. A rota entre Docker no WSL e o motor no Windows está descrita em [observabilidade](docs/engineering-observability.md).

## Executar no WSL

Use o ambiente Conda `deriv-api` com Python 3.13.12. Configure `.env` a partir de `.env.example` com as credenciais da Deriv e da infraestrutura.

```bash
make app-install
make docker-up
python app/train.py
python app/scripts/operations/check_dl_checkpoint.py
python app/run.py
```

O launcher de treino em WSL é `app/scripts/wsl/launch-train-wsl.sh`. O motor usa o checkpoint produzido pelo treino separado (`online_training=false`). Para QA, execute `make app-pre-commit-run` no WSL; o pipeline inclui Vulture com confiança 100, lint, segurança, testes e cobertura de 100% em `app/src`.

## Indicador integrado e múltiplos índices

O indicador observacional fica em [`indicator/`](indicator/README.md) e usa somente o WebSocket público. Seus dashboards ficam em `localhost:3001`; o Grafana do motor automático permanece em `localhost:3000`. Apenas sinais **M5 aprovados** da última vela fechada entram no consenso de execução. Sinais M15/H1 continuam observacionais. Sem modelo TCN próprio, sinal aprovado, dados frescos ou concordância de lado, o ativo não recebe ordem. O motor não converte sinais sintéticos em recomendações para Forex.

```bash
make indicator-up
make app-train-all
make docker-up
make app-run
```

`app-train-all` treina primeiro os modelos públicos, descobre automaticamente os índices M5 aprovados e treina a TCN e os classificadores do motor para esses ativos. Reinicie `app-run` após a aprovação de novos índices para assinar seus streams. A coleta do indicador prioriza índices com modelo M5 aprovado. O motor pode comprar índices diferentes dentro do mesmo ciclo M5: envia as propostas em sequência para reservar saldo e impedir duas compras concorrentes no mesmo ativo; os contratos permanecem abertos simultaneamente. Em recuperação, mantém uma ordem por ciclo. Se o banco do indicador ficar indisponível, o ciclo não reutiliza o último sinal.

O Grafana do motor mostra contratos ativos confirmados com consenso e a view `contract_consensus_outcomes`, baseada exclusivamente em liquidações do broker. O dashboard do indicador mostra sinais aprovados atuais e o catálogo separado marca modelos experimentais como `SEM SINAL`.

## Documentação

- [Arquitetura e componentes](docs/arquitetura.md)
- [Treino e checkpoint TCN](docs/engineering-deep-learning.md)
- [Configuração SSOT](docs/engineering-settings-ssot.md)
- [Execução e risco](docs/binary-senior-playbook.md)
- [Infraestrutura Docker](docs/infra-docker.md)
- [Operação e regras para agentes](AGENTS.md)
- [CI/CD e workflows](.github/workflows/WORKFLOWS.md)
