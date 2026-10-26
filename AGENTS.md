# AGENTS.md — Aether Quantum Engine

## Estrategia operacional vigente em 21/10/2026

`pullback_strategy.enabled=true` com `pullback_strategy.variant="senior_institutional"` substitui a selecao direcional por EV e desliga TCN/meta/loss do caminho principal de decisao. Somente sinais M15 com confluencia institucional podem abrir contratos Rise/Fall de 15 minutos (nos indices de acoes OTC a Deriv exige duracao minima de 15 minutos). O setup institucional senior exige 5 pilares:
1. Tendencia estrutural EMA20/EMA50.
2. Recuo da vela anterior ate 0,5 ATR14 da EMA20.
3. Preco favoravel a referencia OHLC da sessao UTC com fechamento contido nas bandas de +/- 1,5 desvio padrao; sem volume, nao e VWAP.
4. Mitigacao de Fair Value Gap (FVG) entre as velas t-2 e t.
5. Hard Gate de Killzones no Horario de Brasilia (BRT / UTC-3): London Open (03h as 08h), New York Overlap (10h as 13h) e London Fix (14h as 16h); ver [`docs/strategy-pullback.md`](docs/strategy-pullback.md).

O `make app-train` diagnostica o setup somente com OHLC da Deriv para o universo completo de 12 indices sinteticos de acoes OTC (replicas dos ETFs mundiais SPY, QQQ, DIA, EWG, FEZ, etc.) e indices de volatilidade, relata horarios e ranqueia pares pelo proxy direcional de velas, sem promover contratos. A ancora configurada e **OTC_SSMI** (Swiss Market Index 20 / EWL), configurado soberanamente em `config/settings.json` e `drift_symbols.py` para execucao direta no `make app-run`; H1/D1 nao podem gerar ordens. TCN/meta/loss ficam restritos a `make app-research-ml`, sem autoridade operacional.

A DEMO testa ate 200 setups com cotacao valida e stake base de 1% da banca disponivel, sem filtro de EV; apenas liquidacoes `broker` da versao nova contam. REAL exige limite inferior unilateral de 95% positivo, posterior Beta(0,5;0,5), EV positivo e margem de 0,01 na proposta final. A recuperacao usa o payout liquido cotado, com teto de 5% por ordem em DEMO e REAL e reserva por conta. Kelly, D'Alembert e Martingale nao dimensionam esta rota. Sinais e aprovacoes de versoes antigas nao autorizam a estrategia nova. As notas historicas abaixo descrevem o fluxo legado e nao prevalecem sobre esta secao.

Ponto de entrada para agentes Cursor/LLM neste repositorio.

## Idioma e ambiente

- Respostas e commits em **PT-BR**
- Terminal/scripts: **WSL Linux** (nunca CMD/PowerShell nativo)
- Runtime: motor **Python 3.13.12** + **asyncio** no **host** (Conda `deriv-api`); Compose unico com perfis `core,telemetry,indicator`; SSOT runtime [`docs/engineering-python-313-runtime.md`](docs/engineering-python-313-runtime.md)
- Arquitetura: **DDD / hexagonal** — [`docs/engineering-architecture-senior.md`](docs/engineering-architecture-senior.md) + skill `aether-architecture-senior`
- Sem emojis em codigo, logs ou docs tecnicos
- Sem comentarios no codigo (docstrings OK)

## Universo operacional

Rise/Fall CALL/PUT e o unico fluxo operacional ativo. O motor utiliza a rota institucional M15 com o ativo configurado (**OTC_SSMI**); ativos candidatos sao avaliados no `make app-train` com OHLC da Deriv. Componentes legados de ML (meta e loss classifiers) foram integralmente descontinuados e removidos da stack.

- Politica de mercado vigente: matriz de confluencia do setup `senior_institutional` com Hard Gate de Killzones BRT, referencia OHLC da sessao e FVG mitigado; contrato em `docs/engineering-indicator-gates.md`.
- Universo operacional atual: **OTC_SSMI** (Swiss Market Index 20 / Deriv OTC); o diagnostico contempla os 12 ativos mundiais: OTC_SPC, OTC_NDX, OTC_DJI, OTC_GDAXI, OTC_FTSE, OTC_N225, OTC_SX5E, OTC_SSMI, OTC_FCHI, OTC_AEX, OTC_HSI e OTC_AS51.
- Relogio: indicador operacional **900 s** (M15, contrato de 15 minutos fixo); stream micro/MINI legado em 300 s sem autoridade de compra; janela de entrada de ate 60 segundos apos a abertura da vela M15.
- SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`.
- Killzones fixas em Brasilia: 03h-08h, 10h-13h e 14h-16h; o perfil historico nao altera esse gate.
- Cotacao final: `require_quote_edge=true`; probabilidade ausente ou nao finita bloqueia antes da proposta; calcular EV com payout liquido confirmado. A rota institucional M15 aprovada exige EV positivo com stake fixa de 1% da banca disponivel; payout ausente ou invalido bloqueia a execucao.
- Recovery institucional: stake base 1% da banca, amortizacao 1/1 com teto maximo de 5% por ordem, sem multiplicadores geometricos suicidas.

## O que o LLM e / nao e

- **E:** copiloto de engenharia e auditoria
- **Nao e:** decisor de CALL/PUT em runtime (governado exclusivamente pelo setup institucional M15 pullback_strategy.enabled=true)

Doutrina: [`docs/llm-trading-doctrine.md`](docs/llm-trading-doctrine.md)  
Matriz 100% cobertura: [`docs/agent-coverage.md`](docs/agent-coverage.md)  
Rules/skills versionadas: [`.cursor/rules/`](.cursor/rules/) e [`.cursor/skills/`](.cursor/skills/)

## Proibicoes globais

- `force_trade_every_cycle=true` como “fix” de EXEC_EMPTY
- Revenge sizing apos LOSS; “operar mais para aprender” com N baixo
- Remover caps de stake, fila de settlement ou timeouts “temporariamente”
- Introduzir stop loss ou teto acumulado de perda sem mandato do operador
- Arquivos `app/src/**/*.py` acima de **300 linhas**
- Cobertura de testes em `app/src` abaixo de **100%**
- Assunto de commit em ingles; escopo fora do enum commitlint

## Arquivo Histórico / Legado (Não Aplicável à Estratégia Pullback Vigente)

Nota histórica de versões anteriores (**Volatility 75 (1s) M5 / TCN / Meta / Loss-Classifier**): pipeline histórico TCN CALL se Cal ≥0.5 senão PUT → FLIP loss-clf se auto_learn (exit 2, n_train>=1) → Kelly / cover_l0. Esta rota foi integralmente substituída pela `pullback_strategy.enabled=true` descrita na primeira seção deste documento. TCN e ML residem exclusivamente em `make app-research-ml` para pesquisa.

## Escopos commitlint

`all`, `api`, `app`, `config`, `deps`, `domain`, `engine`, `infra`, `llm`, `orchestrator`, `pres`, `release`, `repo`, `risk`, `scripts`, `test`, `tools`, `ws`

Formato: `tipo(escopo): assunto em PT-BR` + corpo obrigatorio.

## Pre-commit

`.pre-commit-config.yaml` → `clean_workspace.py --area --stage` (python/docker/shell; JSON e YAML em steps `Python | JSON *` / `Python | YAML *`; crash-first lint→validate→security→test→build); commitlint primeiro; commit-msg: commitlint.

## Leitura por tarefa

| Tarefa | Abrir primeiro |
|--------|----------------|
| Qualquer mudanca | este arquivo + `docs/agent-coverage.md` |
| Arquitetura DDD / host / event loop / sidecars | `docs/engineering-architecture-senior.md` + skill `aether-architecture-senior` |
| Runtime CPython 3.13 / GC / GIL / Tier2 | `docs/engineering-python-313-runtime.md` + skill `aether-python-313-runtime` |
| Asyncio TaskGroup / starvation | `docs/engineering-python-313-runtime.md` + skill `aether-asyncio-supervisor` |
| Polars / Arrow zero-copy | `docs/engineering-python-deps.md` + skill `aether-polars-arrow` |
| Torch CUDA / to_thread | `docs/engineering-deep-learning.md` + skill `aether-torch-cuda-infer` |
| asyncpg / Timescale ingestao | `docs/infra-docker.md` + skill `aether-asyncpg-timescale` |
| Redis hiredis / ZSET settlement | `docs/engineering-settlement.md` + skill `aether-redis-hiredis` |
| DevOps / CloudOps (Compose/Redis/TS/MinIO) | `docs/engineering-devops-cloudops-senior.md` + skill `aether-devops-cloudops` |
| CALL/PUT/SKIP senior | `docs/binary-senior-playbook.md` + skill `aether-binary-senior` |
| Gates por indicadores (catalogo / backlog) | `docs/engineering-indicator-gates.md` + rule `aether-execution-gates.mdc` + skills `aether-binary-senior` / `aether-session-review` |
| Risco / logs de sessao | doutrina + skill `aether-session-review` |
| Knob em settings | `docs/engineering-settings-ssot.md` + skill `aether-settings-change` |
| Ciclo / warmup | `docs/engineering-orchestrator.md` + skill `aether-cycle-debug` |
| Scale vision / raw_extreme | `docs/engineering-orchestrator.md` + playbook + skills `aether-cycle-debug` / `aether-binary-senior` |
| Settlement | `docs/engineering-settlement.md` + skill `aether-settlement-debug` |
| DL / treino / vies de classe | `docs/engineering-deep-learning.md` + skill `aether-dl-train` |
| Docker / Redis | `docs/infra-docker.md` + skill `aether-infra-stack` |
| Endurecimento Compose / Redis / Timescale / MinIO | `docs/engineering-devops-cloudops-senior.md` + skill `aether-devops-cloudops` |
| Launch-train / sanitize / telemetria | `docs/structure.md` §Scripts + skill `aether-ops-runbook` |
| Deriv PAT/WS | `docs/deriv-api-aether.md` + skill `aether-deriv-connect` |
| QA / pre-commit | `docs/engineering-standards.md` + `.github/workflows/WORKFLOWS.md` + skill `aether-precommit` |
| Deps Python / requirements | `docs/engineering-python-deps.md` + skill `aether-python-deps` (WS max_size/ping, httpx singleton, Polars, MinIO `to_thread`) |
| Higienizacao do repositorio | `docs/engineering-repo-hygiene.md` + skill `aether-repo-hygiene` |
| Fechamento de mudanca (sync superficie) | `docs/engineering-surface-sync.md` + skill `aether-surface-sync` |
| Scaffold / contrato de engenharia | `prompt-model.md` + skill `aether-surface-sync` |
| Volatility 75 (1s) Index / Sinteticos | `docs/deriv-indices-algorithm.md` + rule `aether-v75-market.mdc` + skill `aether-v75-market-analyst` |
| Sizing Single-Strike 4.31% / Payout 0.85 | `docs/medallion.md` + rule `aether-risk-sizing.mdc` + skill `aether-session-review` |
| Verificador de Sinais M15 | `docs/binary-senior-playbook.md` + `docs/engineering-indicator-gates.md` + rule `aether-execution-gates.mdc` + skill `aether-binary-senior` |

Inventario de modulos: [`docs/structure.md`](docs/structure.md)  
Arquitetura runtime: [`docs/arquitetura.md`](docs/arquitetura.md)  
Arquitetura senior (host/DDD/ML/infra/QA): [`docs/engineering-architecture-senior.md`](docs/engineering-architecture-senior.md)

## Integração do indicador na main

A `main` inclui `indicator/` como subsistema de mercado publico, isolado do cliente autenticado. Ele usa `active_symbols` e `ticks_history` em M15/H1/D1; apenas M15 gera propostas no motor de contratos. TCN, meta LightGBM e loss-classifier ficam em pesquisa. A DEMO seleciona o piloto e a REAL seleciona apenas a versao promovida por contratos confirmados do broker. Cada proposta usa 15 minutos, exige vela fechada atual e entrada ate 60 segundos da abertura. A stake base e 1% da banca disponivel em ambas as contas. A recuperacao usa payout cotado e teto de 5% por ordem, sem forcar lado ou setup. Cotacao REAL requer EV positivo e margem de 0,01 acima do break-even apos haircut. Sem sinal atual ou evidencia do broker, nao force nem reutilize lado anterior.

`broker_ev_promotion=true` exige em REAL pelo menos 200 contratos DEMO liquidados pelo broker na mesma versao do setup e limite inferior unilateral de 95% para retorno medio acima de zero. O piloto DEMO pode coletar ate 200 compras por versao com stake base de 1% e recuperacao limitada a 5% por ordem; EV negativo e permitido apenas nesse piloto. O proxy do treino nao significa vantagem economica. O EV REAL usa o payout liquido cotado. Sem evidencia suficiente, REAL nao compra.
`app-train` gera apenas diagnostico por proxy de velas. O ranking nao altera a versao ativa nem a ancora operacional; uma mudanca da implementacao cria nova versao e exige piloto DEMO proprio.

`make docker-up` sobe uma unica stack Compose com indicador, TimescaleDB, Prometheus e Grafana compartilhados. `make app-train` garante essa stack e diagnostica o setup M15/H1/D1 sem promover contratos; TCN, meta e loss permanecem em `make app-research-ml`. Dentro de um ciclo, propostas M15 para ativos distintos sao enviadas sequencialmente com saldo reservado; contratos podem permanecer abertos simultaneamente. No Grafana em `localhost:3000`, a pasta **Aether** mostra compras confirmadas M15 e **Sinteticos** mostra sinais observacionais de todos os periodos. Audite desempenho por versao do setup e liquidacao do broker. Execute `make app-lint` e `make app-test` para validar motor e indicador.

Cada ordem reserva conta/ativo/vela M15 localmente e com `SET NX` no Redis antes do envio. A compra confirmada entra no estado e no risco antes da assinatura de liquidacao. Stream, fila Redis e `profit_table` deduplicam por `contract_id`; a reconciliacao nao deve retirar o contrato antes de chamar o processador comum.
Timeout ou desconexao durante `buy` torna o resultado desconhecido; manter quarentena do ativo por 900 s no Redis e consultar portfolio/`profit_table` antes de operar novamente.

Configuracao operacional atual: ancora OTC_SSMI (Swiss Market Index 20 / EWL); o treino contempla os 12 ativos mundiais de acoes OTC (OTC_SPC, OTC_NDX, OTC_DJI, OTC_GDAXI, OTC_FTSE, OTC_N225, OTC_SX5E, OTC_SSMI, OTC_FCHI, OTC_AEX, OTC_HSI e OTC_AS51) avaliados com OHLC publico da Deriv. A rota do indicador opera com a variante `senior_institutional` (referencia OHLC da sessao, FVG e Hard Gate de Killzones no Horario de Brasilia). Se a Deriv rejeitar explicitamente um ID com `Unknown contract proposal`, somente a rota do indicador tenta uma compra por parametros no mesmo WSS, uma vez, com preco maximo igual a stake e revalidacao da vela M15.

Em `engine_mode=train`, a lista explicita `deep_learning.train_symbols` define o stream e os buffers antes do fetch historico; nao reutilizar a lista menor de ativos operacionais. A descoberta de modelos aprovados e desativada apenas no processo de treino. Cobrir R_10 e R_25 na inicializacao para evitar espera repetida em `0/25000`.

No early stopping TCN, nitidez maior isolada nao reinicia a paciencia quando acuracia/perda de validacao do checkpoint principal nao melhoram. Checkpoint tecnico exportado com val_acc proxima de 0,5 nao e evidencia de previsao util.
