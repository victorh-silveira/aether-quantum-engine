# AGENTS.md — Aether Quantum Engine

Ponto de entrada para agentes Cursor/LLM neste repositorio.

## Idioma e ambiente

- Respostas e commits em **PT-BR**
- Terminal/scripts: **WSL Linux** (nunca CMD/PowerShell nativo)
- Sem emojis em codigo, logs ou docs tecnicos
- Sem comentarios no codigo (docstrings OK)

## Universo operacional

- Universo operacional: **1HZ75V** (Volatility 75 (1s) Index / Deriv)
- Relogio: micro/MINI **900 s** (M15); macro **86400 s** (D1, 365 velas diarias); ciclo/cadência **120 s** (2m); TCN estima deslocamento em **N=1 vela M15** com lookback **30** alinhado ao contrato ops **fixo 15 m (M15)** (`label_horizon_bars=1`, `risk_management.params.duration=15`, `duration_unit="m"`). Rotulagem: **triple_barrier** (Triple Barrier Method: Upper/Lower Log-Vol Barriers + Vertical Expiry Barrier).
- SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`
- Artefactos/treino com granularity/lookback/horizon ≠ settings sao invalidos (gate fail-closed); apos mudar TF/horizonte, retreinar TCN+meta e `make docker-rebuild`
- Treino DL em velas diarias (D1 com 365 barras de historico), elegendo modelo assertivo com **settle_wr** ≥ be+0.03 ou acc ≥ 0.53; deploy reformulado priorizando Edge real vs Breakeven.
- Runtime: `online_training` **false** — DEMO usa checkpoint TCN do `launch-train` (sem retreino deferido no settle); loss-clf e meta `/learn` a cada trade (rebuild containers ml apos mudar env)
- Runtime: `orchestrator.execution.invert_exec_side` **false**. Payout base mercado real: **0.85** (85%). Sizing Kelly: projetado para atingir **4,31% da banca em tacada única M15** (`compounding_rate_daily = 0.0431`, `stop_win_kelly_cycles_target = 1`, `stop_win_kelly_min_fraction = 1.0`, `stop_win_kelly_max_fraction = 1.0`, `max_stake_pct = 0.05`). Ao bater a meta de 4,31% (equivalente a 3% ao dia em 21 dias úteis compostos), encerra a sessão imediatamente com STOP_WIN. Soft Kelly em fusão EV fraca, anti-loss com microestrutura balanceada em barras de 15m.

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

Nota operacional (**Volatility 75 (1s) M15** + arquitetura continua): micro/mini em 900s; pipeline: SCALE → **fusao EV** → **loss-clf FLIP** → **anti-loss com microestrutura M15** (EMA slope 9/21 em 15m; RSI momentum: CALL vetado se $RSI < 0.38$, PUT vetado se $RSI > 0.62$; confirmação de janela `ops_window_bars = 3`) → **neg_edge** com trava de pânico Z-score bilateral ($Z < -2.0$ para CALL / $Z > +2.0$ para PUT gerando `SKIP:NEG_EDGE_ZSCORE_PANIC`). Sizing: tacada única para 4,31% da banca $\text{Stake} = \frac{\text{Meta}}{\text{Payout}} = \frac{0.0431 \times \text{Banca}}{0.85} \approx 5,07\% \to \text{cap } 5,0\%$. Sem revenge sizing pós-loss.

## Escopos commitlint

`all`, `api`, `app`, `config`, `deps`, `domain`, `engine`, `infra`, `llm`, `orchestrator`, `pres`, `release`, `repo`, `risk`, `scripts`, `test`, `tools`, `ws`

Formato: `tipo(escopo): assunto em PT-BR` + corpo obrigatorio.

## Pre-commit

`.pre-commit-config.yaml` → `clean_workspace.py` stages: lint, test (cov 100%), security, cleanup; commit-msg: commitlint.

## Leitura por tarefa

| Tarefa | Abrir primeiro |
|--------|----------------|
| Qualquer mudanca | este arquivo + `docs/agent-coverage.md` |
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
| Launch-train / sanitize / monitores | `docs/structure.md` §Scripts + skill `aether-ops-runbook` |
| Deriv PAT/WS | `docs/deriv-api-aether.md` + skill `aether-deriv-connect` |
| QA / pre-commit | `docs/engineering-standards.md` + skill `aether-precommit` |
| Deps Python / requirements | `docs/engineering-python-deps.md` + skill `aether-python-deps` |
| Higienizacao do repositorio | `docs/engineering-repo-hygiene.md` + skill `aether-repo-hygiene` |
| Fechamento de mudanca (sync superficie) | `docs/engineering-surface-sync.md` + skill `aether-surface-sync` |
| Scaffold / contrato de engenharia | `prompt-model.md` + skill `aether-surface-sync` |
| Volatility 75 (1s) Index / Sinteticos | `docs/deriv-indices-algorithm.md` + rule `aether-sp500-market.mdc` + skill `aether-sp500-market-analyst` |
| Sizing Single-Strike 1% / Payout 0.85 | `docs/medallion.md` + rule `aether-risk-sizing.mdc` + skill `aether-single-strike-risk` |
| Verificador de Sinais & Microestrutura M15 | `docs/binary-senior-playbook.md` + rule `aether-execution-gates.mdc` + skill `aether-m15-signal-verifier` |

Inventario de modulos: [`docs/structure.md`](docs/structure.md)  
Arquitetura runtime: [`docs/arquitetura.md`](docs/arquitetura.md)
