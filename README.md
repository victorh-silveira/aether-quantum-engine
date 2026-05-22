# Aether Quantum Engine 2.0 (Medallion)

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono no framework **Medallion** (estilo Renaissance Technologies): o mercado é um sistema de **sinais ruidosos**, não narrativas macro. O par **`frxEURUSD`** atua como marcapasso de liquidez global; índices US e EU são alvo preditivo em horizonte de **15 minutos**, com modelos **HMM**, **StatArb PCA** e decisão via **Google Gemini** com guardrails quantitativos.

Documentação: [metodologia](docs/medallion.md) | [arquitetura](docs/arquitetura.md)

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

1. Clone e configure o `.env` (tokens Deriv e `GEMINI_API_KEY`).
2. Instale dependências: `pip install -r requirements.txt`.
3. Inicie o motor: `python run.py`.
