# Aether Quantum Engine

[![CI/CD](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](.python-version)
[![Cobertura de testes: 100%](https://img.shields.io/badge/cobertura%20testes-100%25-2E7D32)](pyproject.toml)
[![Polars](https://img.shields.io/badge/dados-Polars-CD792C)](requirements.txt)
[![Docker Compose](https://img.shields.io/badge/infra-Docker%20Compose-2496ED?logo=docker&logoColor=white)](infra/docker/docker-compose.yml)
[![TimescaleDB](https://img.shields.io/badge/series%20temporais-TimescaleDB-FDB515)](infra/docker/docker-compose.yml)
[![Redis](https://img.shields.io/badge/cache-Redis-DC382D?logo=redis&logoColor=white)](infra/docker/docker-compose.yml)
[![Grafana](https://img.shields.io/badge/dashboards-Grafana-F46800?logo=grafana&logoColor=white)](docs/engineering-observability.md)

Motor quantitativo de alta performance para negociacao automatizada de contratos Rise/Fall em indices de acoes mundiais OTC da corretora Deriv.

O fluxo de execucao operacional e governado pelo [Setup Institucional M15](docs/strategy-pullback.md) (`senior_institutional`) com duracao de 15 minutos (900 segundos) e ativo ancora configurado **`OTC_SSMI`** (Swiss Market Index 20).

---

## Destaques de Arquitetura e Engenharia

- **Host CPython 3.13:** Motor assincrono nativo no host sob Linux WSL (Conda `deriv-api`), maximizando a vazao de pacotes WebSocket e eliminando sobrecargas de virtualizacao de rede.
- **DDD Hexagonal (Ports & Adapters):** Dominio puro sem I/O; portas e adaptadores desacoplados para integracao com WebSockets, Redis e TimescaleDB.
- **Polars SSOT:** Processamento colunar exclusivo via Polars com interoperabilidade Apache Arrow zero-copy. Banimento terminante de pandas.
- **Harness Universal de Agentes de IA (`.agents/`):** 24 regras operacionais (`.agents/rules/`), 25 habilidades com runbooks (`.agents/skills/`), personas de subagentes (`.agents/agents/`) e integracao MCP (`.agents/mcp.json`).
- **Garantia de Qualidade:** 100% de cobertura obrigatoria de testes unitarios em `src/` e `src/indicator/`, linters rigorosos (Ruff, Interrogate, Vulture) e limite maximo estrito de 300 linhas por arquivo.

---

## O Setup Institucional Sênior M15 (5 Pilares)

1. **Tendência Estrutural:** Medias moveis exponenciais EMA20 e EMA50 alinhadas e ascendentes/descendentes.
2. **Retração Técnica:** Recuo da vela anterior contido em ate $0,5 \times \text{ATR}_{14}$ da EMA20, com confirmacao fechando a favor da tendencia.
3. **Preço de Sessão UTC:** Fechamento favoravel a media de precos tipicos da sessao UTC e contido dentro das bandas de $\pm 1,5 \sigma$ (sem volume; referencia OHLC pura).
4. **Fair Value Gap (FVG):** Mitigacao confirmada de desequilibrio estrutural entre as velas $t-2$ e $t$.
5. **Hard Gate de Killzones BRT:** Execucao permitida exclusivamente durante:
   - London Open: 03h00 as 08h00 BRT
   - New York Overlap: 10h00 as 13h00 BRT
   - London Fix: 14h00 as 16h00 BRT

---

## Gestão de Risco e Sizing

- **Stake Base:** Exatamente 1% do saldo disponivel da banca.
- **Recuperação Linear 1/1:** Amortizacao linear via payout liquido cotado, com teto maximo inviolavel de 5% por ordem.
- **Veto Econômico:** Exigencia de EV positivo com margem minima de 0,01 acima do break-even na cotacao final. Payout ausente ou degradado aborta a ordem no gate.
- **Banimento Estrito:** Martingale, D'Alembert e Kelly exponencial estao terminantemente proibidos da rota operacional.

---

## Execução Rápida via WSL Linux

### 1. Pré-requisitos
- WSL Linux com Python 3.13.12 (Conda `deriv-api`).
- Docker e Docker Compose instalados.
- Arquivo `.env` configurado a partir de `.env.example`.

### 2. Comandos Principais do Makefile
```bash
# Subir infraestrutura (Redis, TimescaleDB, MinIO, Grafana, Prometheus)
make docker-up

# Instalar dependencias no ambiente Conda deriv-api
make app-install

# Executar diagnostico quantitativo de velas
make app-train

# Iniciar o motor em conta configurada (DEMO ou REAL)
make app-run

# Executar suite completa de linters (Ruff, docstrings, vulture, linhas)
make app-lint

# Executar testes automatizados com verificacao de cobertura 100%
make app-test

# Executar testes do indicador
make indicator-test
```

---

## Documentação Técnica Completa

- [Estratégia Pullback M15](docs/strategy-pullback.md)
- [Playbook de Execução Sênior](docs/binary-senior-playbook.md)
- [Doutrina de Trading para Agentes/LLM](docs/llm-trading-doctrine.md)
- [Matemática Financeira e Sizing](docs/medallion.md)
- [Arquitetura Hexagonal e DDD](docs/engineering-architecture-senior.md)
- [Runtime CPython 3.13 e Asyncio](docs/engineering-python-313-runtime.md)
- [Matriz de Cobertura de Agentes (100%)](docs/agent-coverage.md)
- [Infraestrutura Docker e Containers](docs/infra-docker.md)
- [Padrões de Engenharia e Qualidade](docs/engineering-standards.md)
