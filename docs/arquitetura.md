# Arquitetura — Aether Quantum Engine

Motor assíncrono para trading na Deriv com decisão por **Deep Learning** (TCN, LSTM ou GRU) no índice **`1HZ75V`** (Volatility 75 (1s)). Metodologia quantitativa: [`medallion.md`](medallion.md). Inventário de módulos: [`structure.md`](structure.md). Infra Docker: [`infra-docker.md`](infra-docker.md).

---

## 1. Visão geral

| Aspecto | Valor atual (`config/settings.json`) |
|---------|--------------------------------------|
| Símbolos | `1HZ75V` (âncora `1HZ75V`) |
| Granularidade OHLC (DL) | **86400 s** (`data_handler.granularity`; macro D1) |
| Relógio operacional | **300 s** (`data_handler.micro_granularity`; M5) |
| Histórico para treino | 365 barras diárias (`training_history_bars: 365` / `history_bars: 500`) |
| Lookback | **`deep_learning.lookback`** (settings atuais **30**) → tensor **`[1, 30, 34]`** |
| Features TCN | **34** (`FEATURE_DIM` em `dl_feature_build.py`) |
| Features meta GBDT | **43** (`META_FEATURE_DIM` = 14 + 4 micro-vol + 3 cross + 2 flow) |
| Contrato | `RISE_FALL`, duração **5 m** (ops fixo); label TCN **N=1** vela M5 (`quantum_multi_barrier`) |
| Ciclo | **120 s** (`cycle_interval_seconds`) / **300 s** (`signature_boundary_seconds`; sync M5) |
| Execução | `mandatory_trade_each_cycle: false`; `force` off; `invert_exec_side: false`; fusao EV + signal_skip 1.1 + anti-loss M5 |
| Fail-closed | Meta **opcional** nos settings atuais (`require_meta_for_execution: false`); TCN eager/CUDA local |
| Label | `label_mode: quantum_multi_barrier` (barreiras assimetricas + Vertical Expiry; alt. `triple_barrier`) |
| Meta sessão | Stop win **4,31%** (`compounding_rate_daily: 0.0431`); stop loss desativado |

O mercado é tratado como série temporal ruidosa: a TCN estima `P(CALL)` / `P(PUT)` com calibração e threshold adaptativo; o meta-regressor LightGBM estima `predicted_payoff_edge`; o ranking usa `tcn × max(0.1, 1+z)`. A fusão EV pondera votos direcionais e o filtro anti-loss valida inclinação de EMA 9/21 em barras de 5m, RSI momentum e corpo líquido.

**Invariante temporal:** inferências seguem `signature_boundary_seconds` (**300 s**) via `get_data_state_signature()` — alinhado a **300 s** (micro M5) e **86400 s** (macro D1); ratio macro:micro **1:288**.

**Válvula de starvation:** após **6** ciclos consecutivos bloqueados por qualidade, pisos de margem/edge/Z são atenuados. O piso de edge meta relaxa a partir de **8** skips até floor 0.0.

**Calibração e Zona Neutra:** Modo `neutral_zone` gera estritamente `SKIP:NEUTRAL_ZONE` com execução desativada. Se `raw_prob` estiver em extremos calibrados, preserva Edge genuíno.

**Settlement:** janela de tolerância **600 s** com fila de prioridade Redis (`settlement:queue:priority`) e reconciliação passiva (`portfolio`). Pós-trade, `/v1/learn` alimenta meta-classifier e loss-classifier.

---

## 2. Layout e camadas DDD / hexagonal

Doutrina sênior completa (host 3.13, event loop, CUDA, Polars, sidecars, QA): [`engineering-architecture-senior.md`](engineering-architecture-senior.md).

O motor roda no **host** (WSL/Conda) para acesso direto a CUDA e baixa latência de rede; Redis, Timescale, MinIO, meta e loss ficam em Docker.

```
                    ┌─────────────────────────┐
                    │ Presentation / Inbound  │
                    │ (WS Deriv, Rich UI CLI) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │   Application Layer   │
                     │  (Trading/Orchestr.)  │
                     └─────┬───────────┬─────┘
                           │           │
         ┌─────────────────┘           └─────────────────┐
         ▼                                               ▼
┌──────────────────┐                           ┌───────────────────┐
│   Domain Layer   │                           │ Outbound Ports    │
│  (Pure Entities, │                           │ (Interfaces: DB,  │
│ Value Objects,   │                           │  ML Models, WS)   │
│ Invariants, Risk)│                           └─────────┬─────────┘
└──────────────────┘                                     │
                                                         ▼
                                               ┌───────────────────┐
                                               │   Infrastructure  │
                                               │ (asyncpg, Polars, │
                                               │ Redis, Sidecars)  │
                                               └───────────────────┘
```

```
aether-quantum-engine/
├── app/                 # Código de produção + testes + scripts
│   ├── run.py / train.py
│   ├── aether_asyncio.py
│   ├── src/             # módulos Python (DDD/hexagonal)
│   ├── tests/           # espelho DDD; cobertura 100% em src
│   └── scripts/         # operations, monitor, batch
├── config/settings.json # Runtime
├── data/                # state, session_state, dl/
├── docs/
├── infra/docker/        # Redis, Timescale, MinIO, meta-classifier, loss-classifier
└── linters/             # Ruff, Interrogate, Vulture, ≤300 linhas/arquivo

```

| Camada | Pasta | Papel |
|--------|-------|-------|
| Presentation / inbound | `presentation/` + bootstrap WS | Rich CLI, composition root, logs |
| Application | `application/services/` | Orquestração, DL, direção, quality gates, meta via ports |
| Domain | `domain/` | Risco Kelly + Soft Recovery, invariantes, modelos, math (sem I/O) |
| Outbound ports | Protocols na application | Contratos DB, state, ML, market |
| Infrastructure | `infrastructure/` | Deriv WS/REST, asyncpg, Redis, MinIO, sidecars HTTP |

Regra: **domain** não importa application nem infrastructure. Implementações concretas vêm de `infra_factory.create_infra_services`. Event loop: offload de PyTorch/Polars pesado — não bloquear o hot path WS.
---

## 3. Pipeline de runtime

```mermaid
flowchart TD
  RUN[app/run.py] --> ORCH[Orchestrator]
  ORCH --> BOOT[ws_bootstrap + bootstrap_and_validate_models]
  BOOT --> LOOP[run_orchestrator_main_loop]
  LOOP --> TC[run_trading_cycle_if_ready]
  TC --> SIG[get_data_state_signature micro+macro]
  TC --> LOCK[orchestrator_atomic_state_context]
  LOCK --> DL[collect_deep_learning_decisions]
  DL --> LOCAL[PyTorch eager / CUDA local]
  DL --> BUNDLE[prepare_meta_classifier_cross_symbol_bundle]
  BUNDLE --> META[prefetch_meta_payoff_for_decisions]
  META --> RES[resolve_execution_direction]
  RES --> CHK[execution_direction_checks + discordance]
  CHK --> PSIST[persistence flip ou skip]
  PSIST --> EDGE[meta_edge floor dinamico]
  EDGE --> QG[quality_conviction_suspends_cluster]
  QG -->|HARD micro| MICRO[execution_quality_gate_microstructure]
  QG -->|skip| STV[record_quality_guard_cycle_skip]
  QG -->|ok| COL[collect_cluster_orders / execute_cluster]
  COL --> RM[RiskManager.calculate_stake]
  RM --> TH[TradeHandler.buy_with_parameters]
  TH --> SET[process_contract_settlement]
  SET --> RSC[risk_recovery_state + save_full_state]
  SET --> PSC[post_settlement_cycle / stop_win]
```

### 3.1 Bootstrap

1. `app/run.py` carrega `config/settings.json` + PAT (`.env`: `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`) via `aether_asyncio.run`.
2. `Orchestrator.__init__`: `create_infra_services`, `AuthManager`, `WebSocketManager`, `StreamHandler`, `TradeHandler`, `RiskManager`, `StateManager`, `ExecutionManager`.
3. `validate_engine_risk_config` / `RiskPolicy` no boot (`risk_policy.py`).
4. `validate_infra_services` (fail-fast se `infra.enabled`) → `bootstrap_and_validate_models` (quando MinIO/checkpoints ativos):
   - MinIO → `{symbol}.pth` + artefactos locais
   - Sanity TorchScript multi-probe (`torchscript_sanity_probes`) quando aplicavel
   - Load TCN local (eager/CUDA) + schema + stress infer
5. Auth OTP (health PAT com retry em 502/503/504) → `bootstrap_active_session_targets` (meta 2,60%).
6. Streams macro+micro+ticks → watchdog → settlement worker → loop principal.

### 3.2 Ciclo de trading

`trading_cycle_entry.run_trading_cycle_if_ready`:

1. `process_redis_settlement_queue`
2. Guards (`trading_cycle_entry_guards`): stop-win, assinatura, cadência, contratos ativos
3. `prepare_quality_skipped_cycles_counter` (Redis `state:risk:skipped_cycles_counter`)
4. Sob `orchestrator_atomic_state_context`:
   - `collect_deep_learning_decisions`
   - `quality_conviction_suspends_cluster` → skip + incremento starvation **ou**
   - `executor.execute_cluster(decisions)` → reset starvation

**Lock atômico** (`StateManager._state_lock`): inferência + boleta **não** correm em paralelo com liquidação/persistência de risco.

### 3.3 Assinatura micro+macro (prefixos legados m5/m15)

| Componente | Função |
|------------|--------|
| `m5_boundary_epoch()` | Epoch alinhado ao bloco de **300 s** corrente (nome legado) |
| Micro | `m5:{sym}@{epoch}` — relógio **300 s** (M5) |
| Macro | `m15:{sym}@{epoch}` — relógio **86400 s** (D1) |
| Formato | `m5b:{boundary};m5:...;m15:...` |

Cache inválido quando a assinatura muda; sem assinatura nova, o ciclo aguarda sem re-inferir.

---

## 4. Deep Learning

### 4.1 Features (14D)

| Grupo | Dim | Conteúdo |
|-------|-----|----------|
| Microestrutura | 5 | ticks/barra, intervalo, velocidade, aceleração, std diffs |
| Tradicionais | 22 | RSI, BB, ATR, EMAs, MACD, estocástico, CCI, ADX, Williams, CMO, Keltner, etc. |
| Volatilidade | 5 | vol rolling, vol vs alvo, z-score, implied vol, vol_ratio |
| Persistência | 2 | Hurst, variance ratio |

Módulos: `dl_feature_build.py` (séries), `dl_feature_matrix.py` (linhas/matrizes/tensores), `dl_feature_indicators*.py`, `dl_indicator_config.py`, facade `dl_features.py`.

**Config 100% JSON:** períodos, multiplicadores e thresholds de indicadores vivem em `deep_learning.indicators` (`config/settings.json`). O Python só resolve via `resolve_indicator_config` / `load_indicator_config_from_settings` — sem magic numbers de indicador no código. Mudar períodos sem retreinar o TCN altera a distribuição das features (fingerprint inválido até retreino).

**SSOT de knobs de runtime:** thresholds de negócio, timing/infra e parsers DL restantes também vivem só em `config/settings.json` (quality_gate/starvation/recovery_relax, soft_recovery, recovery_state, kelly runtime, loss_protection.disconnect, market_rank.composite, edge_zscore operacional, live_signal_metrics, meta_payoff_veto, regime/force/cross_corr, timeouts orchestrator/meta/stream/history/shadow, aux_regression_weight, calibration bounds, dynamic_threshold clamps, price_zone, side_equilibrium, deploy_gate). Resolvers `resolve_*_config` / `*_from_settings` são fail-closed (chave ausente = `ValueError`). Exclusões: dims de tensor, enums/reason strings, chaves Redis, epsilons `1e-9`/`1e-12`.

Normalização anti-leakage: `fit_norm_stats` **somente** no split de treino (`dl_splits` purged/embargo).

### 4.2 Modelo e labels

| Peça | Path |
|------|------|
| TCN | `dl_tcn.TemporalDirectionClassifier` (+ `regression_head` multi-task) |
| LSTM/GRU | `dl_lstm.py`; `deep_learning.arch` |
| Perda | Focal assimétrica alta vol (`dl_training_epochs._masked_loss`) |
| Labels | `dl_labels.LabelSpec` — `quantum_multi_barrier` / `triple_barrier` / `spot_forward` / `ma_trend` |
| Checkpoint | `data/dl/{symbol}.pth` + TorchScript / MinIO |
| Deploy gate | `dl_deploy_eval` com **`force_local=True`** (avalia modelo em memória) |

Config atual: `arch: tcn`, `lookback: 30`, `label_mode: quantum_multi_barrier`, thresholds **0.62** / **0.38**, `neutral_half_width: 0.0`.

### 4.3 Treino de sessão

- Walk-forward: `dl_symbol_train` → `train_model_walkforward`
- Bootstrap sequencial: `dl_bootstrap_train` / `run_dl_training_session` via `asyncio.to_thread`
- Retreino deferido: `dl_deferred_train` (thread + semáforo)
- CUDA: init no main thread (`dl_device`); uploads MinIO via `loop.call_soon_threadsafe`
- Gate: `training` até `session_trained` (`dl_training_gate`)

### 4.4 Predição

`predict_symbol_decision_async` (`dl_predict_async.py`):

- Inferência eager/CUDA local no host
- Path sync (`dl_predict.py`) usado pelo mini-deploy de treino: `force_local=True` evita conflito de event loop em thread de treino
- Cache por fingerprint do tensor (`dl_predict_cache`)
- Calibração: `dl_calibration_tolerance` — override TCN macro quando raw &gt;0.65 ou &lt;0.35; zona neutra **off** (`neutral_half_width: 0.0`)

---

## 5. Meta-classificador (LightGBM)

### 5.1 Serviço

| Item | Valor |
|------|-------|
| Container | `aether-meta-classifier` (FastAPI), host **8005→8000** |
| Endpoint | `POST /v2/predict_meta` |
| Cliente | `MetaClassifierClient` + `meta_classifier_pool` (rebind por loop) |
| Timeout | 1,0 s; fallback `predicted_payoff_edge=0.0` |
| Artefatos | `infra/docker/meta-models/*.pkl` |
| Execução | Meta **opcional** (`require_meta_for_execution: false`) |

### Loss-classifier (sidecar HTTP)

| Item | Valor |
|------|-------|
| Container | `aether-loss-classifier`, host **8006→8000** |
| Endpoint | `POST /v1/predict_loss`, `POST /v1/learn`, `POST /v1/retrain` |
| Cliente | `LossClassifierClient` + `loss_classifier_pool` |
| Veto | Soft Kelly em `[0.65, 0.90)`; **FLIP** CALL↔PUT se `p_loss >= hard_p_loss_floor` (**0.90**, `veto_ready`; log `LOSS_CLF \|\| FLIP`); seed com p_loss real |
| Artefatos | `infra/docker/loss-models/*.pkl`; `make docker-reset` limpa + seed predictivo (`veto_ready` se n>=ready_n); `docker-rebuild` recarrega sem apagar TCN |

### 5.2 Vetor 23D

```
META_FEATURE_DIM = 14 (TCN) + 4 (micro-vol zscores) + 3 (cross-symbol) + 2 (flow) = 43
```

| Bloco | Features |
|-------|----------|
| Micro-vol | `micro_bid_ask_spread_momentum[_zscore]`, `volatility_shadow_ratio[_zscore]` |
| Cross | `cross_symbol_prob_delta`, `cross_symbol_vol_ratio_diff`, `cross_symbol_rsi_spread` |
| Flow | `micro_tick_acceleration`, `keltner_deviation_ratio` |

Montagem: `dl_predict_telemetry.prepare_meta_classifier_cross_symbol_bundle` → `extract_meta_feature_vector` → `prefetch_meta_payoff_for_decisions`.

**Nota:** o container `aether-meta-classifier` declara `META_FEATURE_DIM = 23` (alinhado ao app). Treino offline e artefato `.pkl` devem usar a mesma dimensao.

### 5.3 Stacking runtime

1. Bundle cross-symbol + telemetria micro **300 s**
2. Prefetch HTTP → `predicted_payoff_edge`
3. `attach_payoff_edge_zscore_metrics` (janela adaptativa 15–45)
4. `apply_meta_regression_edge`: edge > 0 mantém score TCN; edge < −0,15 + squeeze → `trade_score=0.52` (`[D-SQUEEZE]`)
5. Ranking: `market_decision_score = tcn × max(0.1, 1+z)`
6. Veto Z negativo: `meta_payoff_veto_gate` (waiver em recovery crítico)

### 5.4 Treino offline e anti-leakage

Scripts: `train_meta_vector.py`, `train_meta_data.py`, `train_meta_classifier.py`, `train_meta_optuna.py`.

- Alvo: `Y ≈ PnL / Stake`
- Optuna maximiza Information Ratio; constraint Z-Score OOS
- Anti-leakage: proxy de probabilidade via **retorno passado** (não label forward do horizonte de execução)
- Alinhamento temporal por epoch em `train_meta_vector`

---

## 6. Direção, fusao EV e gates (escopo 1.1)

Runtime atual: TCN ancora Cal → SCALE → soft `signal_skip` → **fusao EV** (`execution_direction_fusion`) escolhe CALL/PUT → **loss-clf FLIP** (ref TCN, ultimo) → **neg_edge** (Cal TCN; `fusion_p_eff` nao lava) / Kelly (`fusion_p_eff` so apos o gate) / caps. Nota: `fusion_loss_weight` nao ve o `p_loss` do mesmo ciclo (FLIP apos fusao); sob seed, `loss_bonus` ja e **0**. Quality gate amplo (RSI/price_zone/SIDE_EQ block) permanece **fora** do codigo; starvation/recovery_relax abaixo sao legado de modulos ainda presentes, nao o eixo operacional.

### 6.1 Motor de direção (modular)

`resolve_execution_direction` orquestra:

| Módulo | Papel |
|--------|-------|
| `execution_direction_checks` | Clamps, sniper stubs, discordance (se ligado), price zone prévia |
| `execution_direction_discordance` | Veto RSI/DI + votos (`discordance_veto_enabled`, default **false**) |
| `execution_direction_persistence` | Após 2 losses no mesmo lado: **flip** toxic escape se o oposto estiver livre; senão skip |
| `execution_direction_meta_edge` | Piso dinâmico de edge (`_resolve_meta_edge_floor`) + `_negative_edge_skip` |
| `execution_direction_resolver` | Finalize: meta regression, price zone + `align_or_keep_meta_side`, SIDE_EQ |
| `execution_price_zone_gate` | BUY/SELL; edge meta &gt; 0 pode manter lado contra a zona |
| `side_equilibrium_gate` | Small-N hard skip / large-N soft; toxic escape **preserva** edge positivo |

| Etapa | Comportamento |
|-------|---------------|
| `infer_dl_direction` | TCN: `P(CALL) ≥ pivot` → CALL, senão PUT (thresholds **0.62/0.38**) |
| Meta edge | Refina score / D-SQUEEZE (meta opcional); edge abaixo do piso dinâmico → `meta_negative_edge` |
| Persistence | Flip CALL↔PUT com `side_eq_toxic_escape` **ou** `persistence_guard_skip` |
| Bloqueio absoluto | `deploy_ok=false`, `gate_reason ∈ {data, predict_error, training}` |

### 6.2 Quality gate dual soft + HARD microestrutura

| Portão | Módulo | Critério |
|--------|--------|----------|
| TCN + meta soft | `execution_quality_gate` | Margem/edge com pisos regulares **0.0** nos settings atuais; Dynamic Recovery Relaxation com `linear≥2` e pendente |
| Meta Z-Score | `execution_quality_gate_meta` | Z vs buffer; waiver recovery se edge ∈ [-0.05, 0.04] e Z&gt;0.5 |
| Cluster | `execution_quality_gate_cluster` | `quality_conviction_suspends_cluster` |
| Microestrutura HARD | `execution_quality_gate_microstructure` | ADX / `vol_ratio` / val_accuracy quando limiares &gt; 0 (settings atuais ADX **0.0**) |
| Sniper stubs | `execution_sniper_gates` | Helpers de banda; stubs Hurst/BB retornam `False` |
| Starvation | `execution_quality_gate_starvation` | Limiar **6** ciclos; edge decay a partir de **8**; Convicção Progressiva (−20%/5 skips) |
| Drawdown relax | `execution_quality_gate_drawdown` | `edge_floor` até **-0.55** |
| Calibração | `dl_calibration_tolerance` | Zona neutra **off**; override TCN em raw extremos |
| Loss protection | `execution_loss_protection` | Caps edge/Z 999; margem operacional **0.0** |
| Meta soft | `meta_payoff_regression` | Soft comprime score sob squeeze; sem hard-block do resolve |
| Settlement | `orchestrator_settlement_queue` | Janela **600 s** + orphan cleaner |

Em modo mandatário, o quality guard emite telemetria `QUALITY_GUARD` / `EXECUTION_FLOW` e delega ao mandatory pick em vez de congelar a esteira por soft alone.

### 6.3 Ranking e seleção

- `execution_market_rank.market_decision_score`
- Redirect inter-símbolo: âncora Z < −0,50 → par Z > +0,50 (`try_inter_symbol_zscore_redirect`)
- `execution_collect` / `execution_mandatory_pick` / `execution_symbols`
- Cointegração Drift sob drawdown: `apply_cointegration_redirect`

---

## 7. Execução e settlement

### 7.1 Fases

- **FASE TREINO** — suspende ordens até `session_trained` em todos os símbolos
- **FASE OPERACAO** — `mandatory_trade_each_cycle: false`; lado via TCN + fusao EV + signal_skip 1.1 (sem quality gate amplo)

### 7.2 ExecutionManager

- Stake via `RiskManager.calculate_stake` → `risk_stake_calc.calculate_stake_for_manager`
- `TradeHandler.buy_with_parameters`: RISE_FALL **5 m** (ops fixo; label N=1 vela M5)
- Reconciliação de stake downgrade Deriv (`executed_stake_reconciliation`)

### 7.3 Settlement

```mermaid
flowchart LR
  POC[proposal] --> IQ[asyncio.Queue]
  IQ --> WK[settlement worker]
  WK --> PCS[process_contract_settlement]
  PCS -->|ws offline| ZSET[Redis ZSET settlement:queue:priority]
  PCS -->|ws online| RSC[risk_recovery_state]
  RSC --> SAVE[save_full_state MULTI/EXEC]
```

Pós-liquidação (`post_settlement_cycle`): stop-win fast-path limpa Redis, cancela fila e `graceful_shutdown(fast_path=True)`; senão retry com teto e recovery transparente de deadlock.

Portões neutralizados em modo mandatário (não bloqueiam ciclo): cooldown pós-LOSS, blackout API, Hurst recovery collect, freeze yield, stubs sniper.

---

## 8. Gerenciamento de risco

| Mecanismo | Módulo |
|-----------|--------|
| Sizing | `risk_stake_calc.calculate_stake_for_manager`: EXPLORE→Kelly; RECOVER→Soft Recovery |
| Soft Recovery (RECOVER) | `soft_recovery_policy` + `dlambert_sizing` (`amort_cycles`, `max_safe_stake_pct`) |
| Tags de stake | `EXPLORE_KELLY` / `RECOVER_DAL_Ln` via `emit_cycle_stake_log` |
| Kelly (EXPLORE) | `kelly_base_fraction`, `stake_sizing`; `fraction: 0.08`, tetos 3,5% |
| Side equilibrium (LLN) | `side_equilibrium` / `side_equilibrium_gate` |
| Consensus entropy | `consensus_stake_penalty.consensus_kelly_retention` (`consensus_penalty_enabled: false`) |
| Recovery persistente | `pending_loss` + `consecutive_losses_linear` + `last_loss_stake` |
| Stop win sessão | `StopWinManager` + `compounding_rate_daily: 0.0431` |
| Stop loss | Desativado |
| Policy boot | `RiskPolicy` / `validate_engine_risk_config` |
| Persistence | `execution_direction_persistence` → flip toxic escape / SKIP / FREEZE |
| Side equilibrium | `side_equilibrium_gate` (toxic escape mantém edge positivo) |
| Val accuracy gate | Limiar configurável (settings atuais sem piso hard de 0.63) |

Facade: `domain/risk/risk_manager.RiskManager.calculate_stake`.

**Sessão:** meta = `session_start_balance × 0.0431` (banca ≥ $100) ou **$10** (banca &lt; $100). Log: `SESSAO INICIADA | Alvo de 4.31%: $… | Stop Loss: DESATIVADO`. Reiniciar o processo inicia sessão nova.

---

## 9. Infraestrutura

| Serviço | Porta | Uso |
|---------|-------|-----|
| Redis | 6379 | Estado atômico, settlement ZSET, starvation counter |
| TimescaleDB | 5432 | Ticks/OHLC; correlação |
| MinIO | 9000/9001 | Checkpoints TCN/TorchScript |
| Meta-classifier | 8005 | LightGBM HTTP |
| Loss-classifier | 8006 | Loss-clf HTTP |
| Deriv | WS/REST | Auth PAT, streams, propostas |

Config `infra.*`: `enabled`, `fail_fast`, `redis`, `timescale`, `minio`, `meta_classifier`, `loss_classifier`.

Persistência: `redis_state_pipeline.write_state_bundle` (MULTI/EXEC) — snapshot, risk hash, pending_loss, session keys, skip counters, market_sig, settlement queue.

Watchdog: `AetherWatchdog` reconecta stream se ticks estagnarem (`watchdog_stale_tick_seconds`, **300 s**).

---

## 10. Configuração (`config/settings.json`)

### `data_handler`
`granularity` (**86400**), `micro_granularity` / `mini_granularity` (**300**), `history_bars` / `training_history_bars` conforme settings (treino tipico **365** barras D1), `fetch_count`, `buffer_limit`, rate-limits de histórico.

### `deep_learning`
`arch`, `lookback` (**30**), `train_symbols`, `confidence_*` (**0.62/0.38**), `calibration.*` (`neutral_half_width: 0.0`), `online_training` (**false**), `deploy_gate.*`, `label_mode` + `label_*`, `tcn.channels`, `training_*`, `model_path_template`, `min_edge_execute`.

### `orchestrator` / `orchestrator.execution`
`cycle_interval_seconds` (**120**), `signature_boundary_seconds` (**300**), `exec_empty_retry_seconds` (**120**), `watchdog_stale_tick_seconds` (**300**), `mandatory_trade_each_cycle` (**false**), `invert_exec_side` (**false**), `require_meta_for_execution` (**false**), `scale_vision.fusion_*` + `signal_skip` 1.1, `settlement_tolerance_window_seconds` (**600**), `post_settlement_is_trading_wait_seconds` (**90**), `warm_up_live_data_timeout_seconds`, `broker_handshake_timeout_seconds`, `state_lock_acquire_timeout_seconds`.

### `risk_management`
`kelly.*` (`fraction: 0.08`, explore piso **0.25%**, tetos stop-win Kelly ate **5%**), `soft_recovery.*` (amort **2/3**, cover **1.10**, linear3 **3.5%**), `min_validation_accuracy_gate` (**0.53**), `params.*` (duration **5** m via `ops_contract_duration_minutes`; `label_horizon_bars` **1**, compounding **0.0431**, stake_min, payout_estimate **0.85**), `large_account_stop_win_pct` (**4.31**), `small_account_*`.

### `infra`
Redis/Timescale/MinIO/meta_classifier/loss_classifier URLs e timeouts.

---

## 11. Observabilidade e QA

Marcadores de log frequentes: `QUALITY_GUARD`, `EXECUTION_FLOW`, `REGIME_GUARD`, `D-SQUEEZE`, `META_CLASSIFIER_FALLBACK`, `STOP_WIN`, `SESSAO INICIADA`, `DATA_SIG`.

Pre-commit (`clean_workspace.py --area --stage`):

| Stage | Conteúdo |
|-------|----------|
| lint | Ruff, Interrogate 100%, Vulture, ≤300 linhas/arquivo |
| validate / build | compileall `app/src` |
| test | pytest + cobertura **100%** em `app/src` |
| security | Bandit + pip-audit + Gitleaks |
| JSON / YAML | steps `Python | JSON *` e `Python | YAML *` |
| docker / shell | jobs paralelos no CI; mesmo orquestrador no pre-commit |
| clean | caches locais |

---

## 12. Diagrama condensado do software

```mermaid
flowchart LR
  subgraph ingestao
    WS[WebSocketManager]
    SH[StreamHandler]
    TB[TickBuffer]
    WD[AetherWatchdog]
  end
  subgraph dl
    FEAT[dl_features 14D]
    LOCAL[eager CUDA local]
    PRED[dl_predict]
    TELE[dl_predict_telemetry]
    META[meta 23D GBDT]
  end
  subgraph direcao
    RES[direction_resolver]
    CHK[direction_checks]
    PSIST[persistence flip/skip]
    EDGE[meta_edge floor]
    QG[quality_gate soft+HARD]
  end
  subgraph exec
    COL[execution_collect]
    EM[ExecutionManager]
    TH[TradeHandler RISE_FALL 30s]
  end
  subgraph pos
    ST[settlement + Redis queue]
    RM[RiskManager Kelly+SoftRecovery]
    LOCK[StateManager Lock]
  end
  WS --> SH --> TB
  WD -->|STALE_DATA| SH
  SH --> FEAT --> LOCAL --> PRED --> TELE --> META --> RES --> CHK --> PSIST --> EDGE --> QG --> COL --> EM --> TH
  TH --> ST --> RM
  COL --> LOCK
  ST --> LOCK
```
