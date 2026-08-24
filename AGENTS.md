# AGENTS.md — Aether Quantum Engine

Ponto de entrada para agentes Cursor/LLM neste repositorio.

## Idioma e ambiente

- Respostas e commits em **PT-BR**
- Terminal/scripts: **WSL Linux** (nunca CMD/PowerShell nativo)
- Sem emojis em codigo, logs ou docs tecnicos
- Sem comentarios no codigo (docstrings OK)

## Universo operacional

- Simbolo unico em runtime: **R_10** (Volatility 10)
- Relogio: micro/MINI **60 s** (M1); macro **7200 s**; ciclo/signature **60 s**; TCN estima deslocamento em **N velas** (nao o close da proxima barra). **N=5** alinhado ao contrato ops **fixo 5 m (M5)** (`label_horizon_bars=5`, `risk_management.params.duration=5`). Rotulagem: **supertrend_atr** (SuperTrend + Volatility ATR Band filter). Ratio macro:micro **1:120**
- SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`
- Artefactos/treino com granularity/lookback/horizon ≠ settings sao invalidos (gate fail-closed); apos mudar TF/horizonte, retreinar TCN+meta e `make docker-rebuild`
- Treino DL direto 5m no `launch-train` e `app-train` (`label_horizon_bars=5`, `duration=5`), elege modelo assertivo com **settle_wr** ≥ be+0.03 ou acc ≥ 0.54; deploy reformulado priorizando Edge real vs Breakeven.
- Runtime: `online_training` **false** — DEMO usa checkpoint TCN do `launch-train` (sem retreino deferido no settle); loss-clf e meta `/learn` a cada trade (rebuild containers ml apos mudar env)
- Runtime: `orchestrator.execution.invert_exec_side` **false** (codigo seletivo `ev_call` permanece para experimento pontual). Fusao EV fraca (EV &lt; `fusion_min_edge_execute`) → soft Kelly `fusion_weak_ev_soft_kelly_mult` **0.50**; sob seed com ambos EV &lt; 0 → `fusion_weak_ev_seed_soft_kelly_mult` **0.25**. Kelly ancorado em `fusion_p_eff` do lado escolhido quando `fusion_applied` **e** Cal TCN ja passou `neg_edge`. Recovery: amort **1/1**, cover **1.50**, stake = cover pleno `pending/payout*1.50` (sem exponencial; `f*` so gate em RECOVER); **PEND material nao e forçado a EXPLORE** por Hurst baixo / NEG_EDGE soft / quality (so near-stop freeze ou cover inviavel); linear3 teto **2.5%**; damping stop-win inicio **1.0** (`target_damping_floor` **0.50** + `span` **0.50**) e perto-meta **0.50**. Explore cold-start: `explore_stake_scale_floor` **0.40**. Loss-clf: `flip_waive_on_closed_candle` **false** (sem sync FLIP→vela).

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

Nota operacional (**escopo 1.1** + arquitetura continua R_10): quality gate amplo legado permanece **fora**; `signal_skip` mini/cal = soft Kelly; `waive_margin_on_pending` **false** (margem direcional sem convicção não abre trade mesmo em recovery); pipeline: SCALE → **fusao EV** → **loss-clf FLIP** (ref TCN, ultimo) → **anti-loss com microestrutura estrita** (EMA slope de 3 barras M5: `Preco > EMA9 > EMA21` e `EMA21[-1] >= EMA21[-3]` para CALL / inverso para PUT; zero bypass de vela: discordância M5 gera `live_exec_discord` obrigatório; RSI momentum: CALL vetado se `RSI < 0.40`, PUT vetado se `RSI > 0.60`) → **neg_edge (Hard Veto Irrevogável)** com trava de pânico Z-score bilateral ($Z < -2.0$ para CALL / $Z > +2.0$ para PUT gerando `SKIP:NEG_EDGE_ZSCORE_PANIC`). Cooldown técnico: 2 ciclos ($120\,\text{s}$) ativado após $L_2+$ (`consecutive_losses_linear >= 2`), com transição suave ($0\,\text{s}$) em $L_1$. Liquidação: polling acelerado para 2.0s na `profit_table` em caso de estagnação. **M1 last-bar = log; confirmacao = janela N=5.** **fusao EV** (`scale_vision.fusion_*`, `fusion_w_micro_bar` **0.10**, `fusion_w_tape` **0.45**, `fusion_w_macro` **0.45**, `fusion_meta_ev_weight` **0.10**, `fusion_loss_weight` **0.45**, `fusion_tcn_shrink_near_half` **0.25**, `fusion_block_when_tcn_pos_edge` **true**, `fusion_block_when_tcn_candle_agree` **true**, `fusion_loss_requires_auto_learn` **true**, `fusion_loss_seed_weight_mult` **0.0**, `fusion_min_edge_execute` **0.035**, `ops_window_bars` **5**) escolhe CALL/PUT por argmax EV (se $\max(EV_{call}, EV_{put}) \le 0.0 \to$ `SKIP:FUSION_NEGATIVE_EV` e `EXEC_EMPTY`), com telemetria `[GATES] || FUSION` (`why=tcn_candle_agree` se TCN==janela ops); seed **nao** alimenta loss_bonus da fusao; `fusion_loss_weight` **nao** ve `p_loss` do mesmo ciclo (FLIP apos fusao); **loss-clf** soft na faixa media (`veto_ready`) e **FLIP** CALL↔PUT **relativo ao TCN** se `p_loss >= hard_p_loss_floor` (**0.90**, so com `veto_ready` e `flip_require_auto_learn` **true** — seed so SOFT; `p_ovr`/`seed_discord` nao FLIP com `auto=0`; seed bootstrap devolve **p_loss real**, sem COLD_START neutro; saida bootstrap `LOSS_BOOTSTRAP_EXIT_N` **16** / primeiro fit so com `n>=LOSS_READY_N` **30** / retrain LOSS com `LOSS_MIN_WIN_FOR_LOSS_RETRAIN` **1** (Docker runtime); `collapsed` no `/predict` zera bonus de fusao); **nunca FLIP** se Edge Cal **e** raw_edge do TCN >= floor **e** tape/janela confirmam TCN (`FLIP_BLOCK:tcn_edge`; `flip_waive_tcn_pos_edge_on_discord` se tape ou janela ≠ TCN; Cal+/raw− libera fusao/FLIP); sob seed (`auto=0`) `flip_seed_block_against_closed_candle` bloqueia FLIP contra a janela (`FLIP_BLOCK:seed_candle`) e `flip_seed_waive_edge_min` (**−0.08**) — `p_ovr` nao fura janela; live (`auto=1`) usa `flip_waive_edge_min` (**−1.0**); janela no alvo usa `flip_candle_p_loss_floor` (**0.85**) so se TCN fraco; **chop** = soft Kelly (**0.55**); **neg_edge** é **Hard Veto Irrevogável**: se Edge Cal do lado $< floor$ (**0.0350**) ou Edge Cal $\le 0.0$, emite `SKIP:NEG_EDGE` e `EXEC_EMPTY` (sem relaxamento por vela ou override). Sem revenge sizing pos-LOSS. Vies CALL/PUT: treino + SIDE_EQ soft.

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

Inventario de modulos: [`docs/structure.md`](docs/structure.md)  
Arquitetura runtime: [`docs/arquitetura.md`](docs/arquitetura.md)
