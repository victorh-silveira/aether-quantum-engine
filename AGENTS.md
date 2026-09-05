# AGENTS.md — Aether Quantum Engine

Ponto de entrada para agentes Cursor/LLM neste repositorio.

## Idioma e ambiente

- Respostas e commits em **PT-BR**
- Terminal/scripts: **WSL Linux** (nunca CMD/PowerShell nativo)
- Runtime: motor **Python 3.13.12** + **asyncio** no **host** (Conda `deriv-api`); sidecars Docker `core,ml`; SSOT runtime [`docs/engineering-python-313-runtime.md`](docs/engineering-python-313-runtime.md)
- Arquitetura: **DDD / hexagonal** — [`docs/engineering-architecture-senior.md`](docs/engineering-architecture-senior.md) + skill `aether-architecture-senior`
- Sem emojis em codigo, logs ou docs tecnicos
- Sem comentarios no codigo (docstrings OK)

## Universo operacional

- Universo operacional: **1HZ75V** (Volatility 75 (1s) Index / Deriv)
- Relogio: micro/MINI **300 s** (M5); macro **86400 s** (D1, 365 velas diarias); ciclo/cadência **300 s** (`require_signature_boundary` **true**, abertura M5); TCN estima deslocamento em **N=1 vela M5** com lookback **30** alinhado ao contrato ops **fixo 5 m (M5)** (`label_horizon_bars=1`, `risk_management.params.duration=5`, `duration_unit="m"`). Rotulagem: **quantum_multi_barrier** (barreiras assimetricas + expiry; alternativa `triple_barrier`).
- SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`
- Artefactos/treino com granularity/lookback/horizon ≠ settings sao invalidos (gate fail-closed); apos mudar TF/horizonte, retreinar TCN+meta e `make docker-rebuild`
- Treino DL em velas diarias (D1 com 365 barras de historico), elegendo modelo assertivo com **settle_wr** ≥ be+0.03 ou acc ≥ 0.53; deploy reformulado priorizando Edge real vs Breakeven.
- Runtime: `online_training` **false** — DEMO usa checkpoint TCN do `launch-train` (sem retreino deferido no settle); loss-clf e meta `/learn` a cada trade (rebuild containers ml apos mudar env)
- Runtime: `orchestrator.execution.invert_exec_side` **false**. Payout base mercado real: **0.85** (85%). Sizing Kelly: projetado para atingir **4,31% da banca em tacada única M5** (`compounding_rate_daily = 0.0431`, `stop_win_kelly_cycles_target = 1`, `stop_win_kelly_min_fraction = 1.0`, `stop_win_kelly_max_fraction = 1.0`, `max_stake_pct = 0.05`). Ao bater a meta de 4,31% (equivalente a 3% ao dia em 21 dias úteis compostos), encerra a sessão imediatamente com STOP_WIN. Soft Kelly em fusão EV fraca, anti-loss com microestrutura balanceada em barras de 5m.

## O que o LLM e / nao e

- **E:** copiloto de engenharia e auditoria
- **Nao e:** decisor de CALL/PUT em runtime (TCN + meta + gates/Kelly)

Doutrina: [`docs/llm-trading-doctrine.md`](docs/llm-trading-doctrine.md)  
Matriz 100% cobertura: [`docs/agent-coverage.md`](docs/agent-coverage.md)  
Rules/skills versionadas: [`.cursor/rules/`](.cursor/rules/) e [`.cursor/skills/`](.cursor/skills/)

## Proibicoes globais

- `force_trade_every_cycle=true` como “fix” de EXEC_EMPTY
- Revenge sizing apos LOSS; “operar mais para aprender” com N baixo
- Remover caps de stake, fila de settlement ou timeouts “temporariamente”
- Arquivos `app/src/**/*.py` acima de **300 linhas**
- Cobertura de testes em `app/src` abaixo de **100%**
- Assunto de commit em ingles; escopo fora do enum commitlint

Nota operacional (**Volatility 75 (1s) M5** + arquitetura continua): micro/mini em 300s; ciclo **300 s** (`require_signature_boundary` **true**, abertura M5); pipeline: SCALE → **fusao EV** → **loss-clf FLIP** → **anti-loss M5** (EMA/candle discord → flip microestrutura `anti_loss_allow_candle_flip` **true** **so se** Edge Cal vela >= **0.015**; hybrid `anti_loss_anchor_agree=false` + last ≠ EXEC → soft `live_discord_weak` ou flip se Edge last >= floor; live confirm/weak/RSI soft 0.30/0.70; seed unstamped HARD; `anti_loss_live_exec_candle_enabled` **false**; ancora hibrida; `fusion_p_eff` sync ao EXEC pos-flip) → **neg_edge** (Edge≤0 HARD; Edge < **0.015** HARD `neg_edge_subfloor_hard` incl. subfloor positivo; Soft_SIZE so soft flags com Edge >= floor; Soft_SIZE piso **2.5%** so se Edge >= **0.015** tambem com PEND; Single-Strike so ALLOW; Z-panic HARD). Sizing: piso Kelly **1%** banca; Soft_SIZE elevado **2.5%** com Edge>=0.015; Single-Strike 4.31% ≈ cap 5.0%. Soft recovery: `cover_enabled` **false** (sem amortizacao em massa); caps max_safe 3.5%. Sem revenge sizing. EMPTY Edge≤0 / Edge&lt;floor / `neutral_zone` = processo ok.

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
| Loss-classifier / Docker ml | `docs/infra-docker.md` + skill `aether-infra-stack` |
| Risco / logs de sessao | doutrina + skill `aether-session-review` |
| Knob em settings | `docs/engineering-settings-ssot.md` + skill `aether-settings-change` |
| Ciclo / warmup | `docs/engineering-orchestrator.md` + skill `aether-cycle-debug` |
| Scale vision / raw_extreme | `docs/engineering-orchestrator.md` + playbook + skills `aether-cycle-debug` / `aether-binary-senior` |
| Settlement | `docs/engineering-settlement.md` + skill `aether-settlement-debug` |
| DL / treino / vies de classe | `docs/engineering-deep-learning.md` + skill `aether-dl-train` |
| Sweep horizonte N / promote | `docs/engineering-deep-learning.md` (secao Sweep) + skill `aether-dl-train` |
| Docker / Redis | `docs/infra-docker.md` + skill `aether-infra-stack` |
| Endurecimento Compose / Redis / Timescale / MinIO | `docs/engineering-devops-cloudops-senior.md` + skill `aether-devops-cloudops` |
| Launch-train / sanitize / monitores | `docs/structure.md` §Scripts + skill `aether-ops-runbook` |
| Deriv PAT/WS | `docs/deriv-api-aether.md` + skill `aether-deriv-connect` |
| QA / pre-commit | `docs/engineering-standards.md` + `.github/README.md` + skill `aether-precommit` |
| Deps Python / requirements | `docs/engineering-python-deps.md` + skill `aether-python-deps` (WS max_size/ping, httpx singleton, Polars, MinIO `to_thread`) |
| Higienizacao do repositorio | `docs/engineering-repo-hygiene.md` + skill `aether-repo-hygiene` |
| Fechamento de mudanca (sync superficie) | `docs/engineering-surface-sync.md` + skill `aether-surface-sync` |
| Scaffold / contrato de engenharia | `prompt-model.md` + skill `aether-surface-sync` |
| Volatility 75 (1s) Index / Sinteticos | `docs/deriv-indices-algorithm.md` + rule `aether-v75-market.mdc` + skill `aether-v75-market-analyst` |
| Sizing Single-Strike 4.31% / Payout 0.85 | `docs/medallion.md` + rule `aether-risk-sizing.mdc` + skill `aether-session-review` |
| Verificador de Sinais & Microestrutura M5 | `docs/binary-senior-playbook.md` + rule `aether-execution-gates.mdc` + skill `aether-binary-senior` |

Inventario de modulos: [`docs/structure.md`](docs/structure.md)  
Arquitetura runtime: [`docs/arquitetura.md`](docs/arquitetura.md)  
Arquitetura senior (host/DDD/ML/infra/QA): [`docs/engineering-architecture-senior.md`](docs/engineering-architecture-senior.md)
