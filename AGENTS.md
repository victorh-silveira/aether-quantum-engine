# AGENTS.md — Aether Quantum Engine

## Estrategia operacional vigente em 21/10/2026

`pullback_strategy.enabled=true` com `pullback_variant="senior_institutional"` substitui a selecao direcional por EV e desliga TCN/meta/loss do caminho principal de decisao. Somente sinais M15 com confluencia institucional podem abrir contratos Rise/Fall de 15 minutos (nos indices de acoes OTC a Deriv exige duracao minima de 15 minutos). O setup institucional senior exige 5 pilares:
1. Tendencia estrutural EMA20/EMA50.
2. Recuo da vela anterior ate 0,5 ATR14 da EMA20.
3. Preco favoravel a Session VWAP com fechamento contido nas bandas de dispersao de +/- 1,5 desvio padrao.
4. Mitigacao de Fair Value Gap (FVG) entre as velas t-2 e t.
5. Hard Gate de Killzones no Horario de Brasilia (BRT / UTC-3): London Open (03h as 08h), New York Overlap (10h as 13h) e London Fix (14h as 16h); ver [`docs/strategy-pullback.md`](docs/strategy-pullback.md).

O `make app-train` diagnostica o setup integrando cotacoes reais de mercado via API HTTP (Yahoo Finance v8) para o universo completo de 12 indices sinteticos de acoes OTC (replicas dos ETFs mundiais SPY, QQQ, DIA, EWG, FEZ, etc.) e indices de volatilidade, aprende os melhores horarios em Brasilia e ranqueia os pares por taxa de acerto e EV esperado (payout 85%). O par promovido vigente e **OTC_SSMI** (Swiss Market Index 20 / EWL), configurado soberanamente em `config/settings.json` e `drift_symbols.py` para execucao direta no `make app-run`; H1/D1 nao podem gerar ordens. TCN/meta/loss ficam restritos a `make app-research-ml`, sem autoridade operacional.

A DEMO testa ate 200 setups com cotacao valida e stake base de 1% da banca disponivel, sem filtro de EV; apenas liquidacoes `broker` da versao nova contam. REAL exige limite inferior unilateral de 95% positivo, posterior Beta(0,5;0,5), EV positivo e margem de 0,01 na proposta final. A recuperacao usa o payout liquido cotado, com teto de 5% por ordem em DEMO e REAL e reserva por conta. Kelly, D'Alembert e Martingale nao dimensionam esta rota. Sinais e aprovacoes de versoes antigas nao autorizam a estrategia nova. As notas historicas abaixo descrevem o fluxo legado e nao prevalecem sobre esta secao.

Ponto de entrada para agentes Cursor/LLM neste repositorio.

## Idioma e ambiente

- Respostas e commits em **PT-BR**
- Terminal/scripts: **WSL Linux** (nunca CMD/PowerShell nativo)
- Runtime: motor **Python 3.13.12** + **asyncio** no **host** (Conda `deriv-api`); Compose unico com perfis `core,ml,telemetry,indicator`; SSOT runtime [`docs/engineering-python-313-runtime.md`](docs/engineering-python-313-runtime.md)
- Arquitetura: **DDD / hexagonal** — [`docs/engineering-architecture-senior.md`](docs/engineering-architecture-senior.md) + skill `aether-architecture-senior`
- Sem emojis em codigo, logs ou docs tecnicos
- Sem comentarios no codigo (docstrings OK)

## Universo operacional

Rise/Fall CALL/PUT e o unico fluxo operacional ativo. O motor utiliza a rota institucional M15 com o ativo promovido soberano (**OTC_SSMI**); ativos candidatos sao avaliados no `make app-train` com historico continuo e cotacoes reais via HTTP. TCN M5 (14D), loss-classifier e meta LightGBM residem exclusivamente em `make app-research-ml` para pesquisa quantitativa, sem interferencia em runtime.

- Politica de mercado vigente: matriz de confluencia do setup `senior_institutional` com Hard Gate de Killzones BRT, Session VWAP e FVG mitigado; contrato em `docs/engineering-indicator-gates.md`.
- Universo operacional atual: **OTC_SSMI** (Swiss Market Index 20 / Deriv OTC); o diagnostico contempla os 12 ativos mundiais: OTC_SPC, OTC_NDX, OTC_DJI, OTC_GDAXI, OTC_FTSE, OTC_N225, OTC_SX5E, OTC_SSMI, OTC_FCHI, OTC_AEX, OTC_HSI e OTC_AS51.
- Relogio: micro/MINI **900 s** (M15, contrato de 15 minutos fixo); janela de entrada de ate 60 segundos apos a abertura da vela M15.
- SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`.
- Horarios aprendidos para OTC_SSMI em Brasilia: 05:00 e 06:00 (janela ideal: 05:00 as 07:00 BRT).
- Cotacao final: `require_quote_edge=true`; probabilidade ausente ou nao finita bloqueia antes da proposta; calcular EV com payout liquido confirmado. A rota institucional M15 aprovada exige EV positivo com stake fixa de 1% da banca disponivel; payout ausente ou invalido bloqueia a execucao.
- Recovery institucional: stake base 1% da banca, amortizacao 1/1 com teto maximo de 5% por ordem, sem multiplicadores geometricos suicidas.

## O que o LLM e / nao e

- **E:** copiloto de engenharia e auditoria
- **Nao e:** decisor de CALL/PUT em runtime (governado exclusivamente pelo setup institucional M5 pullback_strategy.enabled=true)

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
| Loss-classifier / Docker ml | `docs/infra-docker.md` + skill `aether-infra-stack` |
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
| Verificador de Sinais & Microestrutura M5 | `docs/binary-senior-playbook.md` + `docs/engineering-indicator-gates.md` + rule `aether-execution-gates.mdc` + skill `aether-binary-senior` |

Inventario de modulos: [`docs/structure.md`](docs/structure.md)  
Arquitetura runtime: [`docs/arquitetura.md`](docs/arquitetura.md)  
Arquitetura senior (host/DDD/ML/infra/QA): [`docs/engineering-architecture-senior.md`](docs/engineering-architecture-senior.md)

## Integração do indicador na main

A `main` inclui `indicator/` como subsistema de mercado publico, isolado do cliente autenticado. Ele usa `active_symbols` e `ticks_history` em M5/M15/H1; apenas M5 gera propostas no motor de contratos. TCN, meta LightGBM e loss-classifier ficam em pesquisa. A DEMO seleciona o piloto e a REAL seleciona a versao ativa do setup. Cada proposta usa 5 minutos, exige vela fechada atual e entrada ate 30 segundos da abertura. A stake base e 1% da banca disponivel em ambas as contas. A recuperacao usa payout cotado e teto de 5% por ordem, sem forcar lado ou setup. Cotacao REAL requer EV positivo e margem de 0,01 acima do break-even apos haircut. Sem sinal atual ou evidencia do broker, nao force nem reutilize lado anterior.

`broker_ev_promotion=true` exige em REAL pelo menos 200 contratos DEMO liquidados pelo broker na mesma versao do setup e limite inferior unilateral de 95% para retorno medio acima de zero. O piloto DEMO pode coletar ate 200 compras por versao com stake base de 1% e recuperacao limitada a 5% por ordem; EV negativo e permitido apenas nesse piloto. O proxy do treino nao significa vantagem economica. O EV REAL usa o payout liquido cotado. Sem evidencia suficiente, REAL nao compra.
Durante um piloto DEMO com compras atribuidas e menos de 200 liquidacoes confirmadas, ou apos promocao pelo broker, `app-train` preserva a versao M5 aprovada ativa e grava o novo ajuste em `indicator/data/indicator-models/candidates/`. A versao nova nao deve reiniciar a amostra do broker nem remover uma promocao valida. `ev_candidatas=0` na validacao continua sendo ausencia de evidencia de EV favoravel.

`make docker-up` sobe uma unica stack Compose com indicador, TimescaleDB, Prometheus e Grafana compartilhados. `make app-train` garante essa stack e diagnostica o setup M5/M15/H1; TCN, meta e loss permanecem em `make app-research-ml`. Dentro de um ciclo, propostas M5 para ativos distintos sao enviadas sequencialmente com saldo reservado; contratos podem permanecer abertos simultaneamente. No Grafana em `localhost:3000`, a pasta **Aether** mostra compras confirmadas M5 e **Sinteticos** mostra sinais observacionais de todos os periodos. Audite desempenho por versao do setup e liquidacao do broker. Execute `make app-lint` e `make app-test` para validar motor e indicador.

Cada ordem reserva conta/ativo/vela M5 localmente e com `SET NX` no Redis antes do envio. A compra confirmada entra no estado e no risco antes da assinatura de liquidacao. Stream, fila Redis e `profit_table` deduplicam por `contract_id`; a reconciliacao nao deve retirar o contrato antes de chamar o processador comum.
Timeout ou desconexao durante `buy` torna o resultado desconhecido; manter quarentena do ativo por 900 s no Redis e consultar portfolio/`profit_table` antes de operar novamente.

Configuracao operacional atual: ancora OTC_SSMI (Swiss Market Index 20 / EWL); o treino contempla os 12 ativos mundiais de acoes OTC (OTC_SPC, OTC_NDX, OTC_DJI, OTC_GDAXI, OTC_FTSE, OTC_N225, OTC_SX5E, OTC_SSMI, OTC_FCHI, OTC_AEX, OTC_HSI e OTC_AS51) integrados com cotacoes reais via API HTTP do Yahoo Finance v8. A rota do indicador opera com a variante `senior_institutional` (Session VWAP, FVG e Hard Gate de Killzones no Horario de Brasilia). Se a Deriv rejeitar explicitamente um ID com `Unknown contract proposal`, somente a rota do indicador tenta uma compra por parametros no mesmo WSS, uma vez, com preco maximo igual a stake e revalidacao da vela M15.

Em `engine_mode=train`, a lista explicita `deep_learning.train_symbols` define o stream e os buffers antes do fetch historico; nao reutilizar a lista menor de ativos operacionais. A descoberta de modelos aprovados e desativada apenas no processo de treino. Cobrir R_10 e R_25 na inicializacao para evitar espera repetida em `0/25000`.

No early stopping TCN, nitidez maior isolada nao reinicia a paciencia quando acuracia/perda de validacao do checkpoint principal nao melhoram. Checkpoint tecnico exportado com val_acc proxima de 0,5 nao e evidencia de previsao util.
