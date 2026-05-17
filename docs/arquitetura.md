# Arquitetura — Aether Quantum Engine (Medallion 8.0)

Este documento descreve a arquitetura técnica do motor de trading, fundamentada em **Domain-Driven Design (DDD)** e execução assíncrona de alta performance.

---

## 1. Fluxo de Execução

O ciclo de vida do motor é orquestrado de forma reativa à nova vela do símbolo âncora:

1.  **Bootstrap**: `run.py` inicializa o ambiente, carrega `settings.json` e dispara o `Orchestrator`.
2.  **Conectividade**: O `WebSocketManager` estabelece túnel persistente com a Deriv, gerenciando autenticação e keep-alive.
3.  **Ingestão de Dados**: O `StreamHandler` mantém buffers circulares em `numpy` para múltiplos timeframes (**D1, H4, H1, M30, M15, M5, M3, M1**), garantindo causalidade estatística.
4.  **Processamento Quant**:
    - Se `llm.enabled`: O `llm_bridge` compõe o contexto analítico. Os indicadores (Hurst, Z-Score, Entropia, Saturação) são calculados em `src/application/services/llm`.
    - **Soberania Gemini**: A IA processa o prompt e emite uma decisão baseada no Valor Esperado (EV+).
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

O motor é projetado para auditoria imediata. Cada decisão da IA gera um snapshot completo no log (`LLM_AUDIT`), permitindo reconstruir o cenário de mercado exato que levou a um CALL ou PUT meses depois.
