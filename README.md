# Aether Quantum Engine 2.0 (Medallion)

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono no framework **Medallion** (estilo Renaissance Technologies): o mercado é um sistema de **sinais ruidosos**, não narrativas macro. O par **`frxEURUSD`** atua como marcapasso de liquidez global; índices US e EU são alvo preditivo em horizonte de **15 minutos**, com modelos **HMM**, **StatArb PCA** e decisão via **Google Gemini** com guardrails quantitativos.

Documentação: [metodologia](docs/medallion.md) | [arquitetura](docs/arquitetura.md) | [estrutura do repo](docs/structure.md)

Layout: `app/` (codigo e testes), `config/`, `docs/`, `linters/` (pre-commit e release). Ver [docs/structure.md](docs/structure.md).

---

## Estratégia Medallion (resumo)

| Camada | Componente | Função |
|---|---|---|
| **Marcapasso** | `frxEURUSD` + Kalman + HMM | Denoising e regime de volatilidade (reversão vs tendência) no EURUSD |
| **Macro** | Confluência transatlântica | Tags `risk_on` / `risk_off` / divergência a partir dos clusters US e EU (M15) |
| **StatArb** | PCA + Z-Score cross-asset | Resíduos de cointegração curto prazo nos índices; boost/cautela de convicção |
| **Micro** | Hurst, Z, Entropia, MTF D1-M1 | Momentum (H>0.55) vs reversão (H<0.45); cap de convicção em ruído |
| **Decisão** | Gemini + guardrails | `EURUSD`, `US_CLUSTER`, `EU_CLUSTER` independentes; FX ref sem ordens |

### Decisão via LLM (Google Gemini)

Controle em `config/settings.json`, bloco `llm`:
- **Modo Ativo**: `llm.enabled: true` (obrigatório; sem modo simples legado).
- **Mandato**: `llm.system_prompt` define o mandato Medallion enviado ao Gemini (fallback em `sovereign_system.py` se vazio).
- **Gate de Convicção**: `llm.min_conviction_execute` (ex.: 0.60).
- **Sinalização**: `EURUSD`, `US_CLUSTER`, `EU_CLUSTER` (CALL/PUT por cluster).
- **Contexto FX (sem ordens)**: `CONTEXTO_FX_REF` para USD/JPY, AUD/USD e NZD/USD conforme tag macro.

Entrada regional: `risk_on` → cluster **US**; `risk_off` → cluster **EU**; divergência → líder; sem operar âncora EURUSD. Dentro do cluster ativo, por padrão opera **1 índice** com melhor Z-Score StatArb (`statarb_index_max_per_cluster`). Parâmetros: `statarb_lookback`, `statarb_z_threshold`, `statarb_hmm_sigma_*`. Detalhes em [`docs/arquitetura.md`](docs/arquitetura.md).

---

## Gerenciamento de Risco: Critério de Kelly

O motor utiliza um dimensionamento de posição baseado no **Critério de Kelly**:
- **Kelly Fracionário**: `fraction: 0.15` (ou conforme configurado).
- **Stop Win**: Travas para contas pequenas e grandes (em porcentagem).

---

## Observabilidade Quant

O sistema utiliza logs de alta densidade para auditoria em tempo real:

- **`LLM_IO`**: Exibe o payload completo de dados enviado para a IA.
- **`LLM_RESPOSTA`**: Rastreia a decisão da IA para cada cluster e a probabilidade geral.

---

## Stack e Engenharia

- **Core**: Python 3.14 + `asyncio` + NumPy (Kalman, HMM, PCA).
- **Infra**: Deriv WebSocket API v3.
- **IA**: Google Gemini (`gemini-2.5-flash` em `settings.json`).
- **Qualidade**: Cobertura de testes de **100%** em `src`; suite `test_medallion_statarb`.

---

## Execução

1. Clone e configure o `.env` na raiz (tokens Deriv e `GEMINI_API_KEY`).
2. Instale dependências: `make install` ou `pip install -r app/requirements.txt -r app/requirements-dev.txt`.
3. Inicie o motor: `python run.py` ou `python app/run.py`.
4. Monitor (opcional): `python app/scripts/monitor/live_monitor.py` ou `app\scripts\batch\launch-all-live.bat`.

## Backtest Medallion (M15)

Backtest walk-forward do pipeline Medallion (macro, HMM, StatArb PCA, cluster exclusivo, selecao de indice).

Pre-requisitos: token Deriv no `.env` (`AETHER_DEMO_TOKEN` ou `AETHER_LIVE_TOKEN`). Modo Gemini exige tambem `GEMINI_API_KEY`.

**Surrogate quant** (rapido, sem API):

```bash
python app/scripts/backtest/medallion_backtest.py --days 14 --output data/backtest/report.json
```

**Gemini** (mesmo `build_symbol_prompt` e guardrails do live; 1 chamada por barra M15):

```bash
python app/scripts/backtest/medallion_backtest.py --mode gemini --days 14 --output data/backtest/report_gemini.json
```

Padrao: **`--gemini-schedule tag_change`** = consulta Gemini quando a tag macro muda (evita sinal desatualizado no dia). Use `--gemini-schedule daily` para 1 chamada por dia de sessao (~4 consultas em 14 dias).

Outros modos: `--gemini-schedule bar --llm-bar-step 5` (cada N velas). Cache em `data/backtest/gemini_cache.jsonl` grava incrementalmente.

Filtros assertivos em `config/settings.json`: `strategy.excluded_symbols` (SPC, FTSE, NDX), `strategy.macro.allowed_execute_tags` (risk_off e divergencias). Live: `llm.refresh_schedule=tag_change` (padrao) reutiliza decisao Gemini enquanto a tag macro nao mudar.

Cada execucao faz **download fresco** na Deriv (sem cache de mercado). Banca **$100**, Kelly + recuperacao, **stop win diario** e **runtime simulado** ate a meta. Flags: `--days`, `--bars N`, `--bankroll`, `--stake`, `--mode quant|gemini`.
