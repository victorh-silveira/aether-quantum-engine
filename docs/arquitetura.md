# Arquitetura — Aether Quantum Engine (Medallion 8.0)

Este documento descreve a arquitetura técnica do motor de trading, fundamentada em **Domain-Driven Design (DDD)** e execução assíncrona de alta performance.

---

## 1. Fluxo de Execução

O ciclo de vida do motor é orquestrado de forma reativa à nova vela do símbolo âncora:

1.  **Bootstrap**: `run.py` inicializa o ambiente, carrega `settings.json` e dispara o `Orchestrator`.
2.  **Conectividade**: O `WebSocketManager` estabelece túnel persistente com a Deriv, gerenciando autenticação e keep-alive.
3.  **Ingestão de Dados**: O `StreamHandler` mantém buffers circulares em `numpy` para múltiplos timeframes. O contexto LLM usa seis camadas: **D1, H4, H1, M15, M5, M1** (macro a micro), configuráveis em `llm.*_granularity_seconds`.
4.  **Processamento Quant**:
    - Se `llm.enabled`: O `llm_bridge` compõe o contexto analítico. Indicadores Hurst, Z-Score, Entropia Shannon, Velocidade, Aceleracao e Sigma (range %) em `src/application/services/llm`, expostos via `MTF_MATRIX`, `INDICADORES` e `LLM_DADOS_NUM`.
    - **Confluência macro transatlântica** (`global_macro_confluence.py`): agrega retornos dos clusters US e EU para classificar `risk_on`, `risk_off` ou divergência. Movimentos exibidos como RISE/FALL; viés EURUSD quantitativo somente CALL ou PUT; `CONTEXTO_FX_REF` usa RISE/FALL para USD/JPY, AUD/USD e NZD/USD (sem execução).
    - **Soberania Gemini**: A IA processa o prompt e emite uma decisão baseada no Valor Esperado (EV+), com guardrails `apply_macro_confluence_guard` em divergência ou conflito forte com o bias macro.
5.  **Gerenciamento de Ordem**:
    - O `ExecutionManager` solicita o dimensionamento ao `RiskManager` (Critério de Kelly com trava de **Stop Win de 3%**).
    - Para contratos de Multiplicadores, o `ExecutionManager` calcula um **Take Profit Dinâmico** baseado na meta restante da sessão.
    - A ordem é disparada via `trade_handler` e monitorada pelo `settlement_utils`.
6.  **Persistência**: O estado global é salvo em `data/state.json` após cada liquidação.

---

## 2. Organização do Código (`src`)

| Camada | Descrição | Componentes Principais |
|--------|-----------|------------------------|
| **Application** | Lógica de negócio e orquestração. | `Orchestrator`, `ExecutionManager`, `LLM Services` |
| **Domain** | Modelos de dados e regras puras. | `MarketData`, `Trade`, `RiskManager` (Kelly Criterion) |
| **Infrastructure** | Implementação técnica e IO. | `WebSocketManager`, `PersistenceManager`, `Handlers` |
| **Presentation** | Interface de saída e logs. | `TerminalLogger`, `Live Monitor` |

### Detalhe: `src/application/services/llm`

Esta é a "caixa preta" quantitativa do projeto, contendo:
- **`indicators`**: Implementação de Hurst, Shannon Entropy e Z-Score.
- **`llm_bridge_guards`**: Lógica de alinhamento multi-disciplinar (Physics/Chemistry/Biology) e guardrails técnicos.
- **`llm_bridge_telemetry`**: Auditoria estruturada e snapshots de IO (HTTP).
- **`llm_decision`**: Integração direta com a API de Thinking Models do Google.
- **`llm_bridge_utils`**: Normalização estrita de sinais e parsing de respostas.
- **`llm_bridge`**: O "ponteiro" principal que orquestra a chamada entre dados e IA.
- **`global_macro_confluence`**: Voto por cluster, tag Risk-On/Off e linha de referência FX.

### Propagação por cluster (`strategy.correlation`)

Com `correlation.enabled`, a âncora `frxEURUSD` define a direção FX; índices US/EU recebem `US_CLUSTER` e `EU_CLUSTER` exclusivamente da LLM (sem fallback quantitativo). Não há cópia por coeficiente de correlação.

| Cenário índices US+EU | EURUSD | Índices US | Índices EU |
|----------------------|--------|------------|------------|
| RISE juntos (Risk-On) | CALL | CALL | CALL |
| FALL juntos (Risk-Off) | PUT | PUT | PUT |
| Divergem | CALL ou PUT (convicção reduzida) | CALL ou PUT independente | CALL ou PUT independente |

---

## 3. Filosofia de Risco

O projeto adota uma postura **anti-frágil**:
- **Zero Martingale**: Proibição estrita de progressão de stake em perdas.
- **Vantagem Estatística**: Operações baseadas exclusivamente em modelos de convicção com EV positivo.
- **Isolamento de Estado**: Cada símbolo opera seu próprio ciclo, evitando contágio de risco sistêmico.

---

## 4. Garantia de Qualidade (QA)

A arquitetura é validada por uma suíte de testes rigorosa:
- **Cobertura**: 100% de cobertura de código na pasta `src`.
- **Integridade**: Testes unitários para cada serviço e testes de integração para o fluxo do orquestrador.
- **Segurança**: Verificação de vulnerabilidades em dependências e segredos no repositório.

---

## 5. Observabilidade

O motor é projetado para auditoria imediata. Cada decisão da IA gera logs `LLM_AUDIT` e, quando habilitado em `llm.log_llm_io_line`, linhas `LLM_IO` com preview do prompt usuario e do system instruction resolvido (SOVEREIGN + `system_prompt`). O dump opcional `llm.log_llm_io_dump_path` grava JSONL com `mtf_matrix`, `macro_confluence`, `macro_sentiment`, `fx_reference_line`, `indicators_numeric_line`, `institutional_pa_bundle` e tokens sniper por ciclo.
