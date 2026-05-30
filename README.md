# Aether Quantum Engine 2.0 (Medallion — sinteticos M1)

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

**Branch:** `feat/synthetic-indices-m5` — indices de volatilidade sintetica Deriv (**R_100** e clusters VOL), contratos **1 minuto**, decisao **Gemini** com inversao contrarian configuravel.

Documentacao: [medallion sinteticos](docs/medallion-synthetic.md) | [arquitetura sinteticos](docs/arquitetura-synthetic.md) | [estrutura do repo](docs/structure.md)

Para mercado OTC transatlantico (`frxEURUSD` + `OTC_*`), use a branch **`main`**.

Layout: `app/` (codigo e testes), `config/`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## Estrategia (resumo)

| Camada | Componente | Funcao |
|---|---|---|
| **Ancora** | `R_100` | Marcapasso entre blocos VOL baixa e alta |
| **Macro** | Confluencia por cluster M1 | `risk_on` / `risk_off` / divergencia / `indefinido` |
| **StatArb** | Z-Score por indice | Selecao do melhor simbolo no cluster ativo |
| **Decisao** | Gemini + guardrails | `US_CLUSTER`, `EU_CLUSTER`; inversao opcional na execucao |
| **Execucao** | RISE/FALL 1m | Ciclo ~45s; breath e spacing pos-liquidacao M1 |

### LLM (Gemini)

- `config/settings.json`, bloco `llm`: `enabled: true`, `min_conviction_execute` (ex.: 0.60)
- `system_prompt`: mandato Medallion para sinteticos M1
- `strategy.correlation.cluster_invert_llm_side: true` — LLM CALL executa PUT no indice escolhido

---

## Comandos

```bash
make install
make test
make run
make backtest ARGS="--mode gemini --days 14"
```

Pre-commit (WSL): `make pre-commit`.
