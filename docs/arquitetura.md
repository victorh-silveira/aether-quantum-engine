# Arquitetura — Aether Quantum Engine

Motor assíncrono para trading na Deriv com decisão exclusiva por **Deep Learning** (TCN, LSTM ou GRU) nos índices **Drift** (`RDBEAR`, `RDBULL`). A metodologia de negócio quantitativa está em [`medallion.md`](medallion.md); este documento descreve o software.

---

## 1. Visão geral

| Aspecto | Valor atual (`config/settings.json`) |
|---------|--------------------------------------|
| Símbolos | `RDBEAR`, `RDBULL` (âncora `RDBULL`) |
| Granularidade OHLC (DL) | **900 s / M15** (`data_handler.granularity`) |
| Relógio operacional | **60 s / M1** (`data_handler.micro_granularity`) |
| Histórico para treino | 15552 barras M15 (`training_history_bars`, ~162 dias) |
| Lookback | 48 barras M15 por sequência (**12 h** de contexto) |
| Features | **34** (`FEATURE_DIM` em `dl_feature_build.py`) |
| Contrato | `RISE_FALL`, duração **60 s** (execução micro M1) |
| Ciclo do orquestrador | 60 s (`cycle_interval_seconds`, virada M1) |
| Decisão | `collect_deep_learning_decisions` |
| Fases | `FASE TREINO` → `FASE OPERACAO` |
| Execução | Seletiva (`mandatory_trade_each_cycle: false`) ou **contínua** (`true`) |

O mercado é tratado como **série temporal ruidosa**: o modelo estima probabilidade de alta; um **motor de direção linear puro** segue estritamente o sinal da TCN (`P(CALL) > P(PUT)` → CALL, caso contrário PUT) e uma **camada de qualidade neutra** apenas confirma que o sinal é matematicamente válido. Não há vetos táticos, inversões de regime nem skip de ciclo: em modo contínuo o motor boleta em toda virada de minuto M1.

---

## 2. Pipeline de dados

O grafo abaixo reflete o pipeline **aprovado no pre-commit**: o subgrafo `direcao` contém apenas o resolver linear e o gate neutro. Módulos purgados (`execution_entropy_adaptive`, `execution_direction_mean_reversion`, `execution_direction_expansion_veto` e afins) **não** participam do fluxo.

```mermaid
flowchart LR
  subgraph ingestao
    WS[WebSocketManager]
    SH[StreamHandler]
    TB[TickBuffer]
    WD[AetherWatchdog]
  end
  subgraph dl
    FEAT[dl_features 34D]
    TRITON[TritonGrpcClient]
    MODEL[TCN ou LSTM/GRU]
    PRED[dl_predict]
    META[aether-meta-classifier GBDT]
  end
  subgraph direcao
    RES[execution_direction_resolver linear puro]
    QG[execution_quality_gate neutro]
  end
  subgraph exec
    COL[execution_collect]
    SEL[execution_symbols]
    EM[ExecutionManager]
    TH[TradeHandler]
  end
  subgraph pos
    ST[settlement_*]
    RM[RiskManager]
    PM[redis_state_pipeline]
    TS[TimescaleDB]
    MO[MinIO]
  end
  WS --> SH
  SH --> TB
  WD -->|STALE_DATA reconnect| SH
  SH --> FEAT --> TRITON --> MODEL --> PRED --> BUNDLE[dl_predict_build cross-symbol]
  BUNDLE --> META --> RES --> QG --> COL --> SEL --> EM --> TH
  TH --> ST --> RM
  ST --> PM
  ST --> TS
  MODEL --> MO
```

### 2.4 Infraestrutura híbrida

Com `infra.enabled: true`, o motor valida Redis, TimescaleDB e MinIO em `localhost` antes do WebSocket (fail-fast). Estado de risco e sessão persistem em Redis via pipeline atômico (`redis_state_pipeline.write_state_bundle`); ticks e barras vão para Timescale; checkpoints DL sincronizam com MinIO mantendo cache local em `data/dl/`.

**Bootstrap de modelos** (`bootstrap_and_validate_models`):

1. Baixa `{symbol}.pth` e `latest_ts.pt` do MinIO.
2. Valida manifest (`feature_dim`, `lookback`, `norm_mean`/`norm_std`).
3. Forward pass multi-probe em TorchScript (`torchscript_sanity_probes`: zeros, extremos, **regime estressado** RSI/CMO/vol).
4. Com `infra.triton.enabled`: espelha artefatos em `infra/docker/triton-models`, recarrega repositório Triton, valida schema HTTP e executa **`verify_triton_stressed_inference_async`** (inferência concorrente sob tensores estressados; fail-fast se NaN/Inf ou probabilidade fora de `[0, 1]`).

**Inferência de produção** via `TritonGrpcClient` (`triton_grpc_client.py`):

- Canal persistente `grpc.aio.insecure_channel` com keepalive (sem handshake repetido a cada ciclo).
- Predições de `RDBEAR` e `RDBULL` em paralelo com `asyncio.gather`.
- **Timeout rigido de 2,0 s** por requisição (`asyncio.wait_for`); se o Triton exceder o limite, dispara `TritonInferenceTimeout` e fallback imediato para TorchScript em cache local (`dl_predict_triton.py`, log `TRITON_TIMEOUT_FALLBACK`), preservando a janela de 60 s do orquestrador.
- Facade em `triton_inference_client.py` para o restante do motor.

**Meta-regressor tabular** (`aether-meta-classifier`, porta host `8005`):

- Container Python 3.13-slim com FastAPI expõe `POST /v2/predict_meta`.
- Artefatos LightGBM (`.pkl`) montados em `infra/docker/meta-models` → `/models`.
- `MetaClassifierClient` (`meta_classifier_client.py`) consulta o serviço com `httpx.AsyncClient`, timeout **1,0 s** e fallback neutro (`predicted_payoff_edge=0.0`) em falha ou timeout — preserva `trade_score` orgânico da TCN.
- Vetor tabular **39D** enviado ao GBDT: **34** features TCN + **3** cross-symbol (`cross_symbol_prob_delta`, `cross_symbol_vol_ratio_diff`, `cross_symbol_rsi_spread`) + **2** de fluxo micro (`micro_tick_acceleration`, `keltner_deviation_ratio`).
- `prepare_meta_classifier_cross_symbol_bundle` (`dl_predict_build.py`) centraliza telemetria micro M1 paralela (`stamp_micro_frame_telemetry`) e anexa spreads cross-symbol (`attach_cross_symbol_features_to_decisions`) **antes** do prefetch HTTP.
- `collect_deep_learning_decisions` chama o bundle e em seguida `prefetch_meta_payoff_for_decisions`; `execution_direction_resolver` aplica `meta_payoff_regression.apply_meta_regression_edge` sobre o `predicted_payoff_edge` retornado pelo regressor.
- Healthcheck nativo Python (`urllib.request`) — sem dependência de `curl` na imagem slim.
- Treino offline: `train_meta_classifier.py` + `train_meta_optuna.py` + `train_meta_vector.py` (Optuna minimiza MAE; `LGBMRegressor` huber; alvo contínuo `Y = PnL_Real / Stake`; sumário com `train_mae` e `target_variance`).

**Spread de convicção cross-symbol** (`meta_classifier_cross_symbol.py`):

| Feature | Fórmula |
|---------|---------|
| `cross_symbol_prob_delta` | `abs(P(CALL)_RDBULL − P(PUT)_RDBEAR)` |
| `cross_symbol_vol_ratio_diff` | `vol_ratio_micro_BULL − vol_ratio_micro_BEAR` (spread linear M1) |
| `cross_symbol_rsi_spread` | `rsi_micro_BULL − rsi_micro_BEAR` (divergência estocástica 60 s) |

Em regimes de drift paralelo (ambos símbolos com scores altos na mesma direção), o spread baixo sinaliza saturação espelhada — insumo do LightGBM para evitar entradas sem viés relativo.

**Resiliência de ingestão** (`watchdog_service.py` + `stream_reconnect.py`):

- `AetherWatchdog` roda como task assíncrona perpétua após bootstrap do stream.
- Monitora `TickBuffer.last_tick_monotonic()`; se a inanição exceder `watchdog_stale_tick_seconds` (padrão 30 s) com WebSocket aparentemente conectado, entra em estado `STALE_DATA`.
- Antes de reconectar: persiste snapshot de risco (`save_full_state`); em seguida `StreamHandler.reconnect_stream` fecha o WS, reabre sessão OTP e reativa subscrições OHLC/tick sem backfill pesado.
- Encerrado no graceful shutdown (`graceful_shutdown.stop_ingestion_watchdog`).

Config em `orchestrator`: `watchdog_enabled`, `watchdog_stale_tick_seconds`, `watchdog_poll_interval_seconds`.

Redis local usa AOF `appendfsync everysec` (`infra/docker/redis.conf`). `make docker-up` aplica `host-prereq.sh` (`vm.overcommit_memory=1` no WSL).

**Persistência pós-settlement** (`save_full_state`): uma transação Redis `MULTI/EXEC` grava snapshot JSON, hash de risco (`consecutive_losses`, `pending_loss`, cooldowns), hash `session:current`, chaves `session:current:start_balance` e `session:current:target_win`, `recovery:skip_counter` e assinatura de mercado — sem round-trips bloqueantes adicionais na thread principal.

---

### 2.1 Bootstrap

1. `app/run.py` carrega `config/settings.json` e PAT do `.env` (`AETHER_DERIV_PAT` + `AETHER_DERIV_APP_ID`).
2. `validate_infra_services` (quando `infra.enabled`) e `bootstrap_and_validate_models` (checkpoint + TorchScript + sanity + Triton).
3. `restore_orchestrator_state` e `AuthManager` abrem sessão REST/WebSocket via OTP PAT.
4. `Orchestrator` instancia stream, risco, executor e persistência.
5. Após autenticação, `bootstrap_active_session_targets` captura banca inicial e define meta de 1% (`session_target_bootstrap.py`).
6. `StreamHandler.start_candle_stream` busca histórico OHLC e assina velas (`style: candles`) e ticks (`style: ticks`).

### 2.2 Buffer e microestrutura

- `buffer_limit` limita velas em memória por símbolo.
- `history_bars` / `training_history_bars` definem recorte para treino e predição.
- `StreamHandler` assina **dois fluxos OHLC** por símbolo: **M15 (900 s)** para tensor DL `[1, 48, 34]` e **M1 (60 s)** para o relógio do orquestrador.
- `TickBuffer` agrega microestrutura apenas no fechamento de barras **M15**.
- `get_data_state_signature()` combina assinatura **M1 + M15** para reavaliar o cenário a cada minuto sem inferência redundante na GPU.

### 2.3 Assinatura de estado de dados

Para evitar inferências duplicadas na virada de vela, o orquestrador usa `get_data_state_signature()`: concatena epoch e OHLC do último candle fechado por símbolo. Se a assinatura for idêntica ao ciclo anterior, o motor aguarda sem reprocessar.

---

## 3. Deep Learning

### 3.1 Labels e features

**Rótulo** (`dl_labels.py`, modo `ma_trend`):

```
Y[i] = 1.0 se média móvel suavizada indica alta na barra i + horizon
horizon = label_horizon_bars (padrão 1)
```

**34 features** (`FEATURE_DIM` em `dl_feature_build.py`):

| Grupo | Dim | Conteúdo |
|-------|-----|----------|
| Microestrutura | 5 | ticks/barra, intervalo médio, velocidade, aceleração, std diffs |
| Tradicionais | 22 | RSI, delta-RSI, BB, ATR, EMAs, MACD, estocástico, CCI, ADX, DI, Williams, CMO, Keltner, ROC-RSI, etc. |
| Volatilidade | 5 | vol rolling, vol vs alvo, z-score, implied vol ratio, vol_ratio short/long |
| Persistência | 2 | Hurst (R/S), variance ratio |

Normalização anti-leakage: `fit_norm_stats` somente no split de treino walk-forward.

### 3.2 Modelo

- Arquitetura: **`tcn`** (padrão), **`lstm`** ou **`gru`** via `deep_learning.arch`.
- Saída: probabilidade bruta de alta (`raw_prob`).
- Checkpoint v4 em `data/dl/{symbol}.pth` + TorchScript `{symbol}_ts.pt` (espelho MinIO `latest_ts.pt`).
- Inferência via `TritonGrpcClient.infer_symbols_concurrent` quando `infra.triton.enabled`; tensor FP32 **`[1, 48, 34]`** (48 barras M15 = **12 h** de contexto).
- `collect_cluster_orders` opera de forma contínua: sem SKIP por qualidade; mandatory pick garante ordem a cada virada M1 quando a TCN entrega sinal válido.

### 3.3 Treino walk-forward

- Splits temporais com embargo (`dl_splits.py`).
- Early stopping pela perda de validação.
- Retreino: bootstrap de sessão, nova vela, rolling, forçado após loss.
- Treino deferido (`dl_deferred_train.py`): thread em background serializada.
- Deploy gate opcional (`dl_deploy_eval.py`): `deploy_ok=false` bloqueia execução.
- Gate de treinamento: símbolo sem treino da sessão recebe `gate_reason: training`.

### 3.4 Predição

`predict_symbol_decision_async` (`dl_predict_async.py` / `dl_predict_triton.py`):

- Sempre `execute=True` quando a predição técnica é bem-sucedida.
- Calcula indicadores, trend (`dl_trend.py`) e enriquece métricas para o resolver.
- `gate_reason=None` após predição OK; bloqueio só em exceção (`predict_error`).
- Thresholds `confidence_call/put` (0.53/0.47) são bases; com `dynamic_threshold.enabled`, flutuam por `bb_width`, `atr_norm` e regime de volatilidade.
- Com Triton ativo: inferência via `infer_symbol_async`; timeout 2 s dispara fallback TorchScript em cache (`TRITON_TIMEOUT_FALLBACK`).
- Grava `calibrated_prob`, `calibrated_edge` e thresholds dinâmicos em metrics para resolver e quality gate.

---

## 4. Direção e qualidade

### 4.1 Motor de direção com edge contínuo do meta-regressor (`execution_direction_resolver.py`)

A direção macro (`dl_direction`) segue a TCN (M15). O **meta-regressor** (M1) refina o gatilho de execução micro de 60 s via expectativa de retorno contínua:

| Etapa | Comportamento |
|-------|---------------|
| `infer_dl_direction` | Direção pré-computada pelo bridge ou `P(CALL) > pivot` → CALL, caso contrário PUT |
| Probabilidade | `calibrated_prob` (fallback `raw_prob`); pivot = média dos thresholds dinâmicos ou `0.5` |
| Regressão GBDT | `MetaClassifierClient` retorna `predicted_payoff_edge` (prefetch ou sync) |
| **Edge positivo** | `predicted_payoff_edge > 0.0`: mantém `dl_direction` e `trade_score` orgânico da TCN |
| **Edge negativo severo + squeeze** | `predicted_payoff_edge < -0.15` com `bb_width < 0.06` ou `micro_tick_acceleration < 0`: `trade_score=0.52`; `meta_squeeze_downgrade=true`; log `[D-SQUEEZE]` |
| Edge negativo leve | Mantém direção TCN e score orgânico |
| `direction_margin` | `\|P(CALL) − P(PUT)\|` — telemetria de clareza |
| `direction_inverted` | Permanece `False` no fluxo de regressão (sem flip binário) |

**Gatilho D-SQUEEZE** (`meta_payoff_regression.py` + `meta_direction_flip.log_d_squeeze_audit`):

| Condição de squeeze | Detecção |
|---------------------|----------|
| Canal Bollinger esmagado | `bb_width < 0.06` (M1, via `micro_indicators` / `indicators`) |
| Desaceleração institucional | `micro_tick_acceleration < 0` (via `flow_features`) |

Em squeeze com edge severamente negativo, o `trade_score=0.52` força o `consensus_stake_penalty` a comprimir a stake ao piso mínimo da API Deriv ($1.00), sem inverter `exec_direction`.

Bloqueio absoluto (`resolve_execution_direction` retorna `None`) apenas em falha técnica: `deploy_ok == False` ou `gate_reason ∈ {data, predict_error, training}`.

A matriz de correlação cross-symbol (`execution_direction_cross_corr`) e o `execution_volatility_booster` permanecem como telemetria/pisos consultivos.

### 4.2 Gate de qualidade neutro (`execution_quality_gate.py`)

O gate foi neutralizado: não há vetos, penalidades nem skip de ciclo. Sua única função é confirmar que existe sinal matematicamente válido.

| Função | Retorno invariável |
|--------|--------------------|
| `passes_execution_quality(metrics, ...)` | `True`; grava `metrics["regime_skip_cycle"] = False` |
| `apply_quality_penalty_to_metrics(metrics, ...)` | `0.0`; grava `metrics["regime_skip_cycle"] = False` |
| `quality_gate_params(config)` | Pisos default `0.0` (`min_direction_margin`, `inverted_min_score`, `min_adx_normal`) |

Consequência: sob `mandatory_trade_each_cycle: true`, o `ExecutionManager` boleta continuamente a cada virada M1 sempre que a TCN retorna sinal válido. Foram removidos por completo `validate_recovery_asymmetric_gate`, `validate_micro_noise_gate`, `validate_micro_boundary_saturation_gate`, as checagens de ADX colapsado/squeeze em random walk e o barramento `UniversalRegimeEvaluator`.

### 4.3 Pool e seleção

- `execution_recovery_gate.cluster_entry_eligible` — bloqueio **somente técnico** + exige `raw_prob` ou direção.
- `execution_direction.build_execution_candidate` — delega ao resolver linear.
- `execution_symbols.select_best_execution_candidate` / `select_mandatory_execution_candidate` — ranking por `market_decision_score`.
- `execution_mandatory_pick` / `execution_entropy_fallback` — fallbacks que garantem ordem no modo contínuo.

---

## 5. Execução

### 5.1 Fases

- **FASE TREINO** — `_training_phase_gate` suspende ordens até `session_trained` em todos os símbolos.
- **FASE OPERACAO** — `collect_cluster_orders` seleciona melhor candidato ou mandatory pick.

### 5.2 ExecutionManager

- Monta ordens com stake de `RiskManager.calculate_stake`.
- Settlement assíncrono; reentrada via `post_settlement_cycle`.
- Contratos via `TradeHandler.buy_with_parameters`: RISE_FALL, **60 s** (M1), com contexto DL em **M15**.
- Após settlement: `save_full_state` persiste bundle atômico no Redis.

---

### 5.3 Pós-liquidação e encerramento atômico

`post_settlement_cycle.py` agenda reentrada após liquidação com fôlego configurável (`post_settlement_breath_seconds`). O fluxo prioriza **stop win** antes de qualquer round-trip Redis ou ciclo de trading:

```mermaid
flowchart TD
  A[run_post_settlement_breath_and_cycle] --> B{pnl_sessao >= target_win?}
  B -->|Sim| C[clear_current_session_redis_keys]
  C --> D[cancel_settlement_queue_fast]
  D --> E[graceful_shutdown fast_path]
  B -->|Nao| F[retry loop pos-liquidacao]
  F --> G{ciclo incompleto 2x?}
  G -->|Sim| H[emergency_save_session_state]
  H --> I[sys.exit 0]
```

| Etapa | Módulo | Comportamento |
|-------|--------|---------------|
| Curto-circuito stop win | `check_session_limits_before_post_settlement` | Checa `pnl_sessao >= target_win` antes de breath/retry |
| Abort Redis | `clear_current_session_redis_keys` | Remove chaves `session:current:*` sem aguardar MULTI/EXEC pendente |
| Cancelamento de fila | `settlement_queue_ops.cancel_settlement_queue_fast` | Cancela worker e drena fila sem `task_done` handshake |
| Shutdown rápido | `graceful_shutdown(fast_path=True)` | Encerra infra sem aguardar tasks fantasmas pós-reconexão |
| Teto de retry | `_post_settlement_incomplete_streak` | Após 2 ciclos incompletos consecutivos, sinaliza deadlock |
| Saída forçada | `orchestrator_run_loop._enforce_post_settlement_deadlock_exit` | Persiste bundle de emergência em `session_state.json` e `sys.exit(0)` |

Settlement assíncrono via `orchestrator_settlement_queue.py`: worker consome fila sem bloquear o loop principal; no fast-path a fila é cancelada e drenada imediatamente.

---

## 6. Gerenciamento de risco

| Mecanismo | Módulo / config |
|-----------|-----------------|
| Kelly fracionário | `kelly_base_fraction.py`, `stake_sizing.py`, `kelly.fraction` — compressão base de 60% em regime normal (`0.0035 → 0.0012`) |
| Target Proximity Damping | `stake_target_proximity.py` + `RiskManager.apply_kelly_target_proximity_damping` — `Stake_Kelly × (0.40 + 0.60 × remaining_target_pct)` |
| Consensus Entropy Penalty | `consensus_stake_penalty.py` — penalidade convexa em `f*` quando ord diverge dos votos; atenua por `di_diff`, `cmo`, `rsi`; piso `stake_min` em baixo consenso |
| Regime Edge Sizing (waiver) | `consensus_stake_penalty.py` — em recovery, `retention = 1.0` para votos unânimes (6×0/0×6) ou `trade_score >= 0.68` |
| Recovery score waiver | `consensus_stake_penalty.py` — com `pending_loss > 0` ou `consecutive_losses_linear > 0` e votos unânimes ou `trade_score >= 0.68`, `retention = 1.0` |
| Recovery financeiro persistente | `risk_recovery_state.py` — `consecutive_losses_linear` **nao reseta** em WIN operacional enquanto `pending_loss > 0`; reset somente quando o drawdown pendente zera por retornos reais |
| Stop win por sessão ativa | `stop_win_target.py` + `session_target_bootstrap.py` — meta = `session_start_balance × compounding_rate_daily`; sem reset por relógio/calendário |
| Encerramento por meta | `check_session_limits_before_post_settlement` + `graceful_shutdown(fast_path=True)` — aborta Redis e fila de settlement antes do shutdown |
| Stop loss | **Desativado** — sem `daily_max_loss_limit` nem disjuntor de perda no motor |
| Martingale Geométrico | `dlambert_sizing.geometric_martingale_stake` — `Stake = Effective_Base × 2^consecutive_losses_linear` com `Effective_Base = max(dlambert_unit_override, U)`; tag `D'ALEMBERT` quando `pending_total > 0` ou `consecutive_losses_linear > 0` |
| Ancoragem em recovery | `risk_stake_calc.py` ignora consensus penalty e piso `stake_min` quando `pending_total > 0`; evita tag `KELLY` e stake sub-dimensionada em estresse |
| Sem circuit breaker | Removidos `dlambert_circuit_breaker`, `MAX_LINEAR_LEVEL`, `MAX_STAKE_U_MULTIPLE`, `MAX_SESSION_DRAWDOWN_U` e o curto-circuito da tag `D'ALEMBERT_CB`; o cálculo prossegue livre na thread principal escalando exponencialmente até recuperar o passivo total |
| Retração de recovery | WIN parcial em recovery: `consecutive_losses_linear = max(1, n-1)`, reduzindo o expoente da curva |
| Super-concordance Kelly | booster desligado em recovery; ativo em Kelly puro com P≥0.75, 6×0, Hurst>0.55 |
| Trava Hurst N2+ | `recovery_hurst_gate.py` |
| Decaimento Hurst acelerado | `recovery_hurst_decay.py` — `recovery_skip_counter` no Redis |
| Cooldown entrada | `risk_cooldown.py` |
| Cooldown por loss no símbolo | `symbol_loss_cooldown.py` |

### 6.1 Sessão ativa e juros compostos

A meta de lucro segue a planilha de juros compostos (`compounding_rate_daily`, padrão **0,01 = 1%**), aplicada **estritamente por instância de processo**:

1. **Boot** (`ws_bootstrap` → `bootstrap_active_session_targets`): lê saldo vivo da Deriv ou override `session_start_balance` em `risk_management.params`.
2. **Cálculo**: `target_win = session_start_balance × compounding_rate_daily` via `StopWinManager.calculate_session_targets`.
3. **Persistência Redis**: `{prefix}:session:current:start_balance` e `{prefix}:session:current:target_win` no pipeline atômico; hash `session:current` com métricas correntes.
4. **Encerramento**: após settlement ou no início do pós-liquidação, `check_session_limits_before_post_settlement` compara `pnl_sessao` com `target_win`; se atingido, fast-path limpa Redis, cancela fila de settlement e chama `graceful_shutdown(fast_path=True)`.
5. **Deadlock pós-liquidação**: se o ciclo incompleto ocorrer 2 vezes consecutivas, `emergency_save_session_state` grava bundle financeiro e o processo encerra com `sys.exit(0)`.
6. **Nova sessão**: reiniciar `run.py` captura novo saldo e recalcula meta — o operador decide quantas sessões executar no mesmo dia civil.

Log de bootstrap: `SESSAO INICIADA | Alvo de 1%: $XX.XX | Stop Loss: DESATIVADO`.

Com `compounding_enabled: false`, o motor usa alvo legado (`small_account_stop_win` / `large_account_stop_win_pct`).

Logs de sizing expõem `pend=$` (drawdown pendente) e `pnl_sess=$` (P&L acumulado da sessão) em cada cálculo de stake.

### 6.2 Sizing defensivo de proximidade de alvo

Política aplicada sobre a stake Kelly bruta em regime **normal** (fora de recovery). Quando `recovery_active` e `consecutive_losses_linear > 0`, o sizing passa integralmente ao Martingale Geométrico `Kelly_base × 2^n`, sem amortecimento de proximidade:

| Etapa | Fórmula / regra |
|-------|-----------------|
| Compressão Kelly base | `fraction_efetiva = fraction × 0.40` (referência `0.0035 → 0.0012`); recovery mantém `fraction` integral |
| Distância relativa à meta | `remaining_target_pct = max(0, (target_win − pnl_sessao) / target_win)` |
| Amortecimento dinâmico | `target_damping = 0.40 + 0.60 × remaining_target_pct` |
| Stake Kelly comprimida | `Stake_Kelly = Stake_Kelly_raw × target_damping` |

Comportamento: no início da sessão (`pnl_sessao = 0`), `target_damping = 1.0` e apenas a compressão Kelly base atua. Com ~90% da meta atingida, o fator cai para **0.46**, reduzindo superexposição nos ciclos finais antes do stop win.

Persistência: `data/session_state.json` (métricas da sessão corrente), Redis (snapshot atômico + chaves `session:current:*`), `data/state.json` (contratos legado).

---

## 7. Orquestrador

`Orchestrator` (`orchestrator/__init__.py`):

1. `setup_trading_session` — autenticação Deriv e WebSocket.
2. `bootstrap_active_session_targets` — captura banca inicial e meta de stop win da sessão.
3. `start_ingestion_watchdog` — monitoramento de inanição de ticks (modo contínuo).
3. A cada vela do âncora ou `cycle_interval_seconds`:
   - `tick_bars_since_train`
   - `collect_deep_learning_decisions` (inferência Triton concorrente com fallback TorchScript)
   - `executor.execute_cluster`
4. Reconciliação periódica de contratos abertos.
5. Após liquidação: `post_settlement_cycle` (fast-path stop win ou retry com teto) → `save_full_state` quando aplicável.

**Graceful shutdown** (`graceful_shutdown.py`):

- `graceful_shutdown(orch, fast_path=False)` — padrão; aguarda cancelamento de tasks.
- `graceful_shutdown(orch, fast_path=True)` — stop win: cancela fila de settlement (`cancel_settlement_queue_fast`), cancela task pós-liquidação sem handshake prolongado, depois `close_infrastructure_connections`.
- `close_infrastructure_connections` encerra watchdog, Triton gRPC, Timescale, Redis e WebSocket; limpa chaves `session:current:*`; excepthook instalado em `run.py`.

**Loop principal** (`orchestrator_run_loop.py`): a cada iteração verifica `_post_settlement_deadlock`; em deadlock confirmado, persiste estado de emergência e força `sys.exit(0)`.

---

## 8. Configuração crítica

| Bloco | Chaves relevantes |
|-------|-------------------|
| `data_handler` | `granularity`, `history_bars`, `fetch_count`, `buffer_limit` |
| `deep_learning` | `arch`, `lookback`, `confidence_*`, `min_val_accuracy`, `deploy_gate` |
| `orchestrator` | `watchdog_*`, `cycle_interval_seconds`, `idle_cycle_watchdog_seconds` |
| `orchestrator.execution` | `mandatory_trade_each_cycle`, `include_anchor_trades`, `recovery_flip_direction_after_loss` |
| `risk_management.kelly` | `fraction`, `consensus_penalty_*`, `penalty_smoothing_*`, `recovery_*` |
| `risk_management.dlambert` | `dlambert_enabled`, `dlambert_unit_override` (base do Martingale Geométrico) |
| `risk_management.params` | `compounding_enabled`, `compounding_rate_daily`, `session_start_balance`, `duration`, stakes |
| `infra.triton` | `enabled`, `grpc_url`, `http_url`, `model_repo_path` |
| `infra.meta_classifier` | `enabled`, `http_url` (porta host `8005`), `timeout_seconds` |
| `trading` | `mode` (`demo` / `live`) |

---

## 9. Camadas de software

| Camada | Módulos principais |
|--------|-------------------|
| Application / DL | `decision_bridge`, `dl_predict_build`, `dl_predict_async`, `dl_predict_triton`, `dl_gating`, `dl_trend`, `dl_cycle_*`, `model` |
| Application / Meta | `meta_classifier_stacking`, `meta_payoff_regression`, `meta_classifier_features`, `meta_classifier_cross_symbol`, `meta_classifier_flow_features`, `meta_direction_flip` (auditoria D-SQUEEZE) |
| Application / Direção | `execution_direction_resolver` (TCN + edge contínuo), `execution_direction_cross_corr` (telemetria), `execution_volatility_booster` (pisos consultivos), `execution_quality_gate` (neutro) |
| Application / Execução | `execution_collect`, `execution_collect_gather`, `execution_market_rank`, `execution_symbols`, `execution_mandatory_pick` |
| Application / Orchestrator | `Orchestrator`, `execution_manager`, `session_target_bootstrap`, `execution_recovery_gate`, `watchdog_service`, `graceful_shutdown`, `settlement_*`, `settlement_queue_ops`, `post_settlement_cycle`, `orchestrator_run_loop` |
| Domain | `trade`, `market_data`, `probability_entropy`, `risk_manager`, `stop_win_target`, `risk_recovery_state`, `consensus_stake_penalty`, `recovery_hurst_gate`, `dlambert_sizing`, `recovery_conviction`, `stake_sizing` |
| Infrastructure | `triton_grpc_client`, `triton_inference_client`, `websocket_manager`, `stream_handler`, `stream_reconnect`, `tick_buffer`, `trade_handler`, `minio_model_store`, `torchscript_sanity`, `redis_state_pipeline` |
| Presentation | `logger`, `log_dedupe` |

---

## 10. Observabilidade

| Ferramenta | Caminho |
|------------|---------|
| Log ao vivo | `logs/engine.log` |
| Monitor Rich | `app/scripts/monitor/live_monitor.py` |
| CI local | `app/scripts/operations/clean_workspace.py` |
| PAT Deriv | `app/scripts/operations/deriv_pat_connect.py` |

Marcadores de log relevantes:

| Marcador | Significado |
|----------|-------------|
| `[C####] MARTINGALE` / `KELLY` | Stake calculada com `pend=$` e `pnl_sess=$` |
| `RISK: WIN operacional` / `Lucro parcial` | WIN sem reset de recovery enquanto `pending_loss > 0` |
| `RISK: Recovery financeiro zerado` | Drawdown pendente extinto; reset de `consecutive_losses` |
| `SESSAO INICIADA` | Bootstrap de meta por sessão ativa (1% composto; stop loss desativado) |
| `TRITON_TIMEOUT_FALLBACK` | Inferência Triton excedeu 2 s; fallback TorchScript local |
| `WATCHDOG: STALE_DATA` | Inanição de ticks; reconexão controlada do stream |
| `CICLO: ciclo pos-liquidacao incompleto` | Retry pós-liquidação; após 2× consecutivas → encerramento forçado |
| `STOP_WIN` / fast-path | Meta da sessão atingida; Redis limpo e shutdown imediato |
| `[D-SQUEEZE]` | Downgrade de score em compressão M1; métricas `bb_width`, `tick_accel`, `predicted_payoff_edge`, `score` |

---

## 11. Garantia de qualidade

- Cobertura **100%** em `app/src` (pytest + coverage).
- Pre-commit: Ruff, Interrogate, Vulture, pylint duplicate-code, máximo **300 linhas** por arquivo em `app/src`.

---

## 12. Referências

- [medallion.md](medallion.md) — princípios quant e perfil de qualidade
- [README.md](../README.md) — execução e pré-requisitos
- [deriv-api.md](deriv-api.md) — API Deriv
- [infra-docker.md](infra-docker.md) — stack Docker, Triton, Redis AOF
- [CHANGELOG.md](CHANGELOG.md) — histórico de releases
