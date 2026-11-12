# AGENTS.md — Aether Quantum Engine

Ponto de entrada unico e soberano para agentes autonomos de IA e desenvolvedores no repositorio Aether Quantum Engine.

---

## 1. Identidade e Papel do Agente

- **O que voce e:** Copiloto senior de engenharia de software, arquitetura de sistemas e auditoria quantitativa.
- **O que voce NAO e:** Decisor discricionario de ordens em runtime. A abertura de posicoes e governada de ponta a ponta pelo setup quantitativo institucional M15 (`pullback_strategy.enabled=true`).
- **Comportamento basico:** Zero alucinacao, rigor analitico, estrita adesao aos contratos de software e verificacao constante atraves de suites de testes automatizadas.

---

## 2. Ambiente Operacional e Diretrizes de Engenharia

- **Idioma oficial:** Todas as respostas, documentacoes tecnicas e mensagens de commit devem ser escritas em **PT-BR**.
- **Terminal e Scripts:** Todas as operacoes devem ser executadas no ambiente **WSL Linux (Bash)**. Nunca utilize comandos nativos do Windows CMD ou PowerShell.
- **Runtime:** Motor executado no host com **CPython 3.13.12** e **asyncio** (ambiente Conda `deriv-api`).
- **Arquitetura de Software:** **DDD (Domain-Driven Design) e Hexagonal (Ports & Adapters)**. O dominio deve permanecer puro, sem imports de infraestrutura ou frameworks.
- **DataFrames e Series:** O **Polars** e o unico padrao para manipulacao colunar de alta performance (`LazyFrame`, `DataFrame`, zero-copy com Apache Arrow). O uso de pandas e estritamente proibido.
- **Integridade Visual:** Sem emojis em codigo-fonte, logs de sistema ou documentacoes tecnicas.
- **Documentacao de Codigo:** Codigo autoexplicativo com docstrings completas. Comentarios soltos no corpo de funcoes devem ser evitados.
- **Contrato de Commits:** Commits devem seguir rigorosamente o padrao Conventional Commits em PT-BR utilizando o script oficial `scripts/wsl/commit.sh` (ou `make app-commit`), garantindo conformidade com pre-commit e commitlint.

---

## 3. Estrategia Quantitativa Operacional Vigente

A autoridade operacional de mercado e governada pelo setup institucional de recuo em 15 minutos:

- **Contrato Operacional:** Rise/Fall (CALL/PUT) com duracao fixa de **15 minutos** (900 segundos).
- **Ativo Ancora Configurado:** **OTC_SSMI** (Swiss Market Index 20 / EWL - Deriv OTC).
- **Estrategia Ativa:** `pullback_strategy.enabled=true` com `pullback_strategy.variant="senior_institutional"`.
- **Janela de Abertura:** A entrada deve ocorrer em ate 60 segundos apos a abertura da vela M15, exclusivamente sob vela anterior fechada e sem gaps.

### A Matriz dos 5 Pilares Institucionais:
1. **Tendencia Estrutural:** Alinhamento de medias moveis exponenciais EMA20 e EMA50.
2. **Recuo Tecnico:** Recuo da vela anterior ate no maximo 0,5 ATR14 da EMA20.
3. **Preco de Referencia de Sessao:** Cotacao contida dentro das bandas de +/- 1,5 desvio padrao em relacao a referencia OHLC da sessao UTC (sem volume, referencia OHLC pura).
4. **Mitigacao de Desequilibrio (FVG):** Confirmacao de mitigacao de Fair Value Gap entre as velas $t-2$ e $t$.
5. **Hard Gate de Killzones no Horario de Brasilia (BRT / UTC-3):**
   - **London Mid:** 07h00 as 08h00 BRT (hora `7`)
   - **Pre-NY Open:** 09h00 as 10h00 BRT (hora `9`)
   - **New York Overlap:** 10h00 as 11h00 e 12h00 as 14h00 BRT (horas `10, 12, 13`)
   - **London Fix / Pos-Fix:** 15h00 as 17h00 BRT (horas `15, 16`)
   - Qualquer vela fora dessas janelas horarias (`[7, 9, 10, 12, 13, 15, 16]`) e sumariamente rejeitada pelo gate.

### Gestao de Capital e Dimensionamento:
- **Stake Base:** 1% do saldo disponivel da banca.
- **Recuperacao Institucional:** Amortizacao linear 1/1 baseada no payout liquido cotado, com teto maximo absoluto de **5% por ordem**, sem multiplicadores geometricos (Martingale, Kelly e D'Alembert estao terminantemente banidos da rota operacional).
- **Gate de Cotacao Real:** Abertura de contratos em conta REAL exige valor esperado (EV) estritamente positivo com margem minima de 0,01 acima do break-even apos corte de haircut. Payout ausente, invalido ou nulo bloqueia a proposta imediatamente.
- **Isolamento de Pesquisa:** Modelos legados de aprendizado profundo (TCN, meta-classificador LightGBM e loss-classifier) residem exclusivamente no modulo de pesquisa experimental (`make app-research-ml`) e estao totalmente desligados do fluxo de decisao operacional.

---

## 4. Subsistema Indicador Publico de Mercado

O diretorio `src/indicator/` compoe o subsistema publico de leitura e analise de mercado:
- Opera desacoplado e sem credenciais de negociacao, consumindo apenas endpoints publicos da Deriv (`active_symbols`, `ticks_history`).
- Monitora os 12 ativos mundiais de acoes OTC (OTC_SPC, OTC_NDX, OTC_DJI, OTC_GDAXI, OTC_FTSE, OTC_N225, OTC_SX5E, OTC_SSMI, OTC_FCHI, OTC_AEX, OTC_HSI, OTC_AS51) e indices de volatilidade.
- Compartilha a infraestrutura Docker Compose da raiz (TimescaleDB, Redis, Prometheus e Grafana).
- Diagnosticos emitidos em `make app-train` utilizam proxies de velas para relatorios estatisticos e nunca promovem contratos nem alteram configuracoes ativas autonomamente.

---

## 5. Proibicoes Globais e Hard Gates

Qualquer violacao dos seguintes pontos resulta em reprovacao imediata nos processos de integracao continua:
1. Ativar `force_trade_every_cycle=true` para maquiar ciclos sem operacao.
2. Modificar tamanhos de lote por impulso ou vinganca apos perda (revenge sizing).
3. Remover caps de protecao de stake, timeouts ou travas da fila de settlement.
4. Introduzir stop loss financeiro arbitrario sem mandato formal do operador.
5. Criar ou manter arquivos em `src/**/*.py` acima do limite de **300 linhas**.
6. Reduzir a cobertura de testes unitarios em `src` abaixo de **100%**.
7. Criar commits com mensagens em ingles ou escopos fora do enum commitlint.

---

## 6. Harness de Agentes de IA (`.agents/`)

As instrucoes executivas para ferramentas de IA estao padronizadas e centralizadas na pasta canonica `.agents/`:
- **Regras Operacionais (Rules):** [`.agents/rules/`](.agents/rules/) (todas configuradas com `alwaysApply: true`).
- **Habilidades de Execucao (Skills):** [`.agents/skills/`](.agents/skills/) (procedimentos operacionais passo a passo).
- **Integracao de Ferramentas (MCP):** [`.agents/mcp.json`](.agents/mcp.json) (servidores de dados, metricas e containers).
- **Universalidade:** Estrutura agnostica compativel nativamente com qualquer ecossistema de agentes IA/LLM atraves do padrao aberto `AGENTS.md` e do harness `.agents/`.

---

## 7. Matriz de Leitura Rapida por Tarefa

| Tarefa de Engenharia | Documento Principal | Regra Obrigatoria | Habilidade Recomendada |
|---|---|---|---|
| Qualquer mudanca no projeto | `AGENTS.md` + `docs/agent-coverage.md` | `aether-engineering.mdc` | `aether-surface-sync` |
| Arquitetura DDD / Host / Hexagonal | `docs/engineering-architecture-senior.md` | `aether-architecture-senior.mdc` | `aether-architecture-senior` |
| Runtime CPython 3.13 / Asyncio / Event Loop | `docs/engineering-python-313-runtime.md` | `aether-python-313-runtime.mdc` | `aether-asyncio-supervisor` |
| Polars / Arrow / Manipulacao de Dados | `docs/engineering-python-deps.md` | `aether-python-deps.mdc` | `aether-polars-arrow` |
| Conexao Deriv / WSS / PAT / OTP | `docs/deriv-api-aether.md` | `aether-deriv-api.mdc` | `aether-deriv-connect` |
| Setup Institucional M15 e Estrategia Pullback | `docs/strategy-pullback.md` | `aether-indicator-consensus.mdc` | `aether-binary-senior` |
| Gates de Execucao e Validacao de Sinais | `docs/engineering-indicator-gates.md` | `aether-execution-gates.mdc` | `aether-binary-senior` |
| Indicador Publico e Sinais Sinteticos | `docs/strategy-pullback.md` + `docs/arquitetura.md` | `aether-indicator.mdc` | `aether-indicator` |
| Risco, Sizing 1% e Gestao de Capital | `docs/medallion.md` + `docs/llm-trading-doctrine.md` | `aether-risk-sizing.mdc` | `aether-session-review` |
| Configuracoes e Knobs SSOT | `docs/engineering-settings-ssot.md` | `aether-settings-ssot.mdc` | `aether-settings-change` |
| Orquestrador, Ciclos e Watchdog | `docs/engineering-orchestrator.md` | `aether-orchestrator.mdc` | `aether-cycle-debug` |
| Liquidacao Redis ZSET e Reconciliacao | `docs/engineering-settlement.md` | `aether-settlement.mdc` | `aether-settlement-debug` |
| Ingestao TimescaleDB e Pool asyncpg | `docs/infra-docker.md` | `aether-infra.mdc` | `aether-asyncpg-timescale` |
| Cache Redis hiredis e Filas | `docs/engineering-settlement.md` | `aether-settlement.mdc` | `aether-redis-hiredis` |
| Infra Docker Compose e Servicos de Sidecar | `docs/infra-docker.md` | `aether-infra.mdc` | `aether-infra-stack` |
| DevOps / CloudOps / Hardening de Containers | `docs/engineering-devops-cloudops-senior.md` | `aether-infra.mdc` | `aether-devops-cloudops` |
| Observabilidade, Telemetria e Logs | `docs/engineering-observability.md` | `aether-logging.mdc` | `aether-session-review` |
| Pesquisa Deep Learning / Treinamento Offline | `docs/engineering-deep-learning.md` | `aether-deep-learning.mdc` | `aether-dl-train` |
| Dinamica de Indices Sinteticos Continuos | `docs/deriv-indices-algorithm.md` | `aether-v75-market.mdc` | `aether-v75-market-analyst` |
| Analise de Amostra Estatistica e Significancia | `docs/sample-size-lln.md` | `aether-sample-size.mdc` | `aether-session-review` |
| Scripts Operacionais e Rotinas de Suporte | `docs/structure.md` | `aether-scripts.mdc` | `aether-ops-runbook` |
| Higienizacao de Codigo e Repositorio | `docs/engineering-repo-hygiene.md` | `aether-repo-hygiene.mdc` | `aether-repo-hygiene` |
| Padrao de Commits e Hook commit-msg | `docs/engineering-standards.md` | `aether-git-commit.mdc` | `aether-git-commit` |
| Garantia de Qualidade, Linters e Testes 100% | `docs/engineering-standards.md` | `aether-testing.mdc` | `aether-precommit` |
| Fechamento de Ciclo e Sincronizacao de Superficie | `docs/engineering-surface-sync.md` | `aether-surface-sync.mdc` | `aether-surface-sync` |
