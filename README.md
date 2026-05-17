# Aether Quantum Engine 2.0 (Medallion 8.0)

[![Python](https://img.shields.io/badge/Python-3.14-3776AB?logo=python&logoColor=white)](.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor de trading quantitativo assíncrono de ultra-convicção focado no mercado de **Multiplicadores (Leveraged Forex)**. O sistema opera em ciclos de 15 minutos (**M15**) no par **EURUSD**, integrando análise multi-disciplinar (Física de Mercados, Química de Saturação e Biologia de Exaustão) via **Google Gemini**.

---

## Soberania Medallion 8.0

O motor evoluiu para o framework **Medallion 8.0**, inspirado nas estratégias de James Simons. A decisão não é apenas técnica, mas uma confluência causal:

| Camada | Métricas Chave | Propósito |
|--------|----------------|-----------|
| **Física** | Hurst (H), Entropia de Shannon | Detectar persistência de tendência e ruído estocástico. |
| **Química** | PH de Saturação, Reatividade | Identificar zonas de exaustão de liquidez e aceleração catalítica. |
| **Biologia** | Fadiga de Momentum, VO2 | Medir a "saúde" da tendência e o consumo de volume. |
| **Financeira** | Z-Score, Volatilidade (Sigma) | Arbitragem de reversão à média e controle de risco. |

### Decisão via LLM (Google Gemini)

Controle em `config/settings.json`, bloco `llm`:
- **Modo Sniper**: Ativado com `llm.enabled: true`.
- **Alavancagem**: Utiliza multiplicador de **x800** para maximizar retornos em pequenas variações.
- **Thinking Budget**: O modelo utiliza **2048 tokens** de raciocínio interno para simular o próximo candle antes de emitir o sinal.
- **Gate de Convicção**: Execução mandatória apenas se `conviction >= 0.55`.
- **Sinalização**: `MULTUP`, `MULTDOWN` ou `SKIP` (em caso de incerteza absoluta ou Sigma > 3.0).

---

## Gerenciamento de Risco: Critério de Kelly

O motor utiliza um dimensionamento de posição determinístico baseado no **Critério de Kelly** com foco em alvos diários:
- **Stop Win Diário**: Meta estrita de **3% da banca** por sessão.
- **Take Profit Dinâmico**: O sistema calcula o lucro restante para a meta e o define como o `take_profit` da ordem.
- **Zero Stop Loss**: Operações sem SL manual, utilizando o *stop-out* natural da stake (risco controlado por Kelly).
- **Kelly Fracionário (0.35)**: Ajustado para atingir a meta diária com alta seletividade (poucas operações).
- **Max Stake Cap**: Limite estrito de $50.00 por operação no EURUSD.

---

## Observabilidade Quant

O sistema utiliza logs de alta densidade para auditoria em tempo real:

- **`LLM_AUDIT`**: Exibe o Regime de Mercado, Sigma e a Linha Quant multi-timeframe (**D1 / H4 / H1 / M30 / M15 / M5 / M3 / M1**).
- **`LLM_RESPOSTA`**: Rastreia a decisão da IA, convicção (probabilidade), latência da API e tokens consumidos.
- **`EXEC`**: Detalha o cálculo do stake via Kelly e o ID da transação.

---

## Stack e Engenharia

- **Core**: Python 3.14 + `asyncio`.
- **Infra**: Deriv WebSocket API v3.
- **IA**: Google Gemini (`google-genai`) com suporte a **Thinking Models**.
- **Matemática**: `numpy` para processamento causal de vetores OHLC.
- **Qualidade**:
  - Cobertura de testes de **100%** em `src`.
  - Linting rigoroso via **Ruff**.
  - Auditoria de segurança (**Bandit**, **pip-audit**, **Gitleaks**).
  - CI/CD completo via GitHub Actions.

---

## Execução

1. Clone e configure o `.env` (tokens Deriv e `GEMINI_API_KEY`).
2. Instale dependências: `pip install -r requirements.txt`.
3. Inicie o motor: `python run.py`.
4. Monitore em tempo real: `python scripts/operations/live_monitor.py`.

Referência da API: [`docs/deriv-api.md`](docs/deriv-api.md).
Arquitetura: [`docs/arquitetura.md`](docs/arquitetura.md).
