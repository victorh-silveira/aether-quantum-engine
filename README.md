# Aether Quantum Engine 2.0 (Medallion 8.0)

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor de trading quantitativo assíncrono focado no mercado de **Índices EUR e USD**, utilizando o par **EURUSD** como âncora macroeconômica. A decisão é tomada via **Google Gemini (LLM)** com direções independentes por cluster (`US_CLUSTER`, `EU_CLUSTER`) e travas quantitativas.

---

## Estratégia: Estrategista Macro Medallion

O motor opera no framework **Medallion**, analisando a transmissão de força do EURUSD para os clusters de índices, agora blindado com regras de física de mercado:

| Camada | Métrica | Propósito / Regra |
|---|---|---|
| **Macro** | Correlação (Âncora) | Define o regime (Sincronia ou Divergência) entre EURUSD e os clusters. |
| **Tendência** | Hurst (H) > 0.55 | Segue a tendência dominante do ativo/cluster e ignora Z-Score esticado. |
| **Reversão** | Hurst (H) < 0.55 | Usa Z-Score para operar reversão à média em mercados sem tendência. |
| **Segurança** | Entropia > 3.5 | Trava de probabilidade máxima em 0.75 para evitar excesso de confiança em ruído. |

### Decisão via LLM (Google Gemini)

Controle em `config/settings.json`, bloco `llm`:
- **Modo Ativo**: `llm.enabled: true`.
- **Gate de Convicção**: Execução apenas se `conviction >= 0.70`.
- **Sinalização**: `EURUSD`, `US_CLUSTER` e `EU_CLUSTER`.

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

- **Core**: Python 3.14 + `asyncio`.
- **Infra**: Deriv WebSocket API v3.
- **IA**: Google Gemini (`gemini-3.1-pro-preview`).
- **Qualidade**: Cobertura de testes de **100%** em `src`.

---

## Execução

1. Clone e configure o `.env` (tokens Deriv e `GEMINI_API_KEY`).
2. Instale dependências: `pip install -r requirements.txt`.
3. Inicie o motor: `python run.py`.
