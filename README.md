# Aether Quantum Engine 2.0

[![Python](https://img.shields.io/badge/Python-3.13.12-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](app/tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono para a Deriv: decisão por **Deep Learning** (TCN/LSTM/GRU) nos índices **Drift** (`RDBEAR`, `RDBULL`), contratos **RISE_FALL** de **120 s** com contexto macro **600 s** (proporção **1:5**), meta-regressor LightGBM (**43D**) de expectativa de retorno contínuo, e **soft recovery** com caps de segurança quando há passivo pendente. As chaves de assinatura ainda usam prefixos legados `m5`/`m15` para os relógios configurados de **120 s** / **600 s**.

A operação divide-se em duas fases: **FASE TREINO** (nenhuma ordem até todos os modelos concluírem o treino da sessão) e **FASE OPERACAO** em **esteira mandatária contínua** (`mandatory_trade_each_cycle: true`): o motor seleciona candidato obrigatório quando o pool DL é tecnicamente válido e aprovado pelo quality gate dual soft (TCN + meta Z-Score) e pelos vetoes HARD de microestrutura. Triton permanece **fail-closed** (`infra.triton.require_for_execution: true`); o meta é **opcional** para execução (`require_meta_for_execution: false`).

Documentação: [arquitetura](docs/arquitetura.md) | [estrutura e módulos](docs/structure.md) | [metodologia quant](docs/medallion.md) | [infra Docker](docs/infra-docker.md) | [Deriv API](docs/deriv-api.md) | [índice docs](docs/README.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` + `TickBuffer` + `AetherWatchdog` | WebSocket Deriv dual-timeframe: OHLC macro **600 s** (assinatura legado `m15`) para DL/regimes + OHLC micro **120 s** (assinatura legado `m5`) para gatilho do ciclo; ticks agregados por barra fechada; watchdog reconecta stream em inanição (>**25 s**) |
| Fases | `_training_phase_gate` | Suspende a operação até todos os modelos concluírem o treino da sessão |
| Predição DL | `decision_bridge` + `dl_predict_*` + TCN | **34 features** TCN; bundle meta **43D**; Triton gRPC timeout **0,50 s**; fail-closed obrigatório para Triton |
| Meta GBDT | `meta_classifier_client` + `aether-meta-classifier` | Regressão tabular **43D**; `predicted_payoff_edge` contínuo (opcional para execução) |
| Z-Score payoff | `payoff_edge_zscore` | Janela adaptativa 15–45; `meta_payoff_edge_zscore` |
| Direção | `execution_direction_resolver` + persistence guard + quality gate | TCN define lado; zona neutra `[0.46, 0.54]`; margem hard `0.03`; persistence **skip** (sem flip); D-SQUEEZE rebaixa score |
| Rotulagem DL | `dl_labels` + `LabelSpec` | Padrão `spot_forward`; `ma_trend` / Triple Barrier via config |
| Quality / starvation | `execution_quality_gate*` | Dual soft TCN+meta; vetoes HARD de microestrutura (`adx_starvation`, `vol_ratio_starvation`, `val_accuracy_gate`); starvation a partir de **6** skips |
| Ranking | `execution_market_rank` | Score `tcn × max(0.1, 1+z)` |
| Execução | `ExecutionManager` + lotes fracionados | Proposta atômica; RISE_FALL **120 s** |
| Risco | `RiskManager` + `soft_recovery_policy` | Kelly `fraction=0.005`, teto **3,5%**, `max_safe_stake_cap`, stop win 2,60%; consensus penalty **desligado** |
| Concorrência | `StateManager` + barreira atômica | Lock serializa inferência, liquidação e persistência |
| Inferência | `TritonGrpcClient` | Canal persistente; rebind por event loop |

Ciclo do orquestrador: `orchestrator.cycle_interval_seconds` / `signature_boundary_seconds` (**120 s**). Contexto DL: `data_handler.granularity` (**600 s**, tensor `[1, 72, 34]`). Contrato: `risk_management.params.duration` (**120 s**, RISE_FALL alinhado ao fechamento micro). Proporção multi-timeframe **1:5** (120:600).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (`RDBEAR`, `RDBULL`; ancora `RDBULL`) |
| `data_handler` | `granularity` (macro **600 s**), `micro_granularity` (**120 s**), `history_bars` / `training_history_bars` (**23328**), `fetch_count`, `buffer_limit` |
| `deep_learning` | `arch`, `lookback` (**72**), `label_mode` (`spot_forward`), calibration (`neutral_half_width: 0.04`), thresholds **0.54/0.46**, `indicator_gating`, `deploy_gate` |
| `orchestrator.execution` | `mandatory_trade_each_cycle`, `require_meta_for_execution: false`, `quality_gate`, settlement **90 s** |
| `risk_management.kelly` | `fraction` (0.005), caps 3,5%, `consensus_penalty_enabled: false`, recovery |
| `risk_management.soft_recovery` | Soft D'Alembert paramétrico (`max_safe_stake_cap`, amort 2–5, `coing_redirect`) |
| `infra` | Redis, Timescale, MinIO, Triton (`infer_timeout_seconds: 0.50`, `require_for_execution`), meta-classifier |

## Ambiente híbrido Docker

O motor (`run.py` / `train.py`) roda no host Conda/WSL. Redis, TimescaleDB, MinIO, **Triton** e **meta-regressor** sobem via Docker em `localhost`:

```bash
make docker-up
```

Pipeline: `host-prereq` → `triton-prereq` → `compose up` (profiles `DOCKER_PROFILES`, padrão `core,gpu,ml`) → wait healthy → `timescale-lifecycle` → `docker-hydrate` → `docker-smoke`. Redis sobe com AOF `everysec`.

| Target | Uso |
|--------|-----|
| `make docker-up` | Stack completa (core+gpu+ml) |
| `make docker-up-core` | Só Redis, Timescale e MinIO |
| `make docker-rebuild` | Rebuild do meta-classifier + up |
| `make docker-smoke` | Valida endpoints da stack |

Compose declara GPU via `gpus: all`. Detalhes em [docs/infra-docker.md](docs/infra-docker.md).

```bash
docker exec -it aether-redis redis-cli CONFIG GET appendonly
docker exec -it aether-redis redis-cli CONFIG GET appendfsync
```

Com `infra.enabled: true`, o startup valida os serviços (fail-fast), sincroniza TorchScript no Triton e executa **sanity estressado** (RSI/CMO/vol extremos) antes do WebSocket Deriv.

Variáveis na raiz (`.env` único — Deriv + infra Docker):

| Variável | Uso |
|----------|-----|
| `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` | Conta Deriv |
| `AETHER_PG_USER`, `AETHER_PG_PASSWORD`, `AETHER_PG_DB` | TimescaleDB |
| `AETHER_MINIO_ACCESS_KEY`, `AETHER_MINIO_SECRET_KEY` | MinIO |
| `AETHER_TRITON_GRPC`, `AETHER_TRITON_HTTP` | Triton Inference Server |
| `AETHER_META_CLASSIFIER_HTTP` | Meta-classificador LightGBM (padrão `http://localhost:8005`) |
| `AETHER_DOCKER_HEALTH_TIMEOUT` | Timeout do wait healthy (padrão `300`) |

Copie `cp .env.example .env` e preencha o PAT. Validação Deriv: `python app/scripts/operations/deriv_pat_connect.py`.

---

## Gerenciamento de risco

- **Kelly fracionário** com win rate dinâmico e tetos `max_stake_pct`.
- **Consensus Entropy Penalty**: presente no código; nos settings atuais `consensus_penalty_enabled: false`.
- **Penalty smoothing em recovery**: com drawdown pendente e `trade_score` alto, waiver de consenso quando a penalidade estiver habilitada.
- **Recovery financeiro persistente**: WIN operacional **não** zera `consecutive_losses` enquanto `pending_loss > 0`; reconciliação de stake downgrade preserva drawdown real.
- **Soft recovery (D'Alembert)** via `soft_recovery_policy`: em recovery, stake `max(U × m(n), cover)` com **passo fixo** `m(n)=1.15` para `n ∈ {3,4}`; demais níveis `factor^n`; teto `max_safe_stake_cap` (**4.20**) e hard floor **5%** se banca &lt; $100; amortização 2–5 ciclos; `coing_redirect_drawdown_threshold` **15.00**.
- **Stop win por sessão ativa**: meta = `banca_inicial × 2,60%` (banca ≥ $100) ou **$10** fixo (banca &lt; $100); `finalize_stop_win_shutdown` purge Redis + log CRITICAL + fast-path.
- **Stop loss desativado**: sem disjuntor de perda diária interno.
- **Lotes fracionados**: stakes acima de `max_single_stake_limit` (padrão $200) divididas em N ordens com proposta atômica por sub-lote; falha técnica de proposta aborta o cluster sem inflar `pending_loss`.
- Cooldown por símbolo após sequência de losses (`symbol_loss_rotation_cycles`): rota o par Drift após loss linear sem pausar o ciclo.
- **Proteção contra loss** (`execution_loss_protection`): `min_direction_margin: 0.03`; caps edge/Z **999**.
- **Gate de acurácia**: `min_validation_accuracy_gate: 0.63` (veto HARD de microestrutura).

---

## Fases, recovery e execução

- **FASE TREINO**: ao iniciar a sessão, todo símbolo retreina pelo menos uma vez. Enquanto qualquer modelo não concluir, nenhuma ordem é enviada.
- **FASE OPERACAO mandatária** (`mandatory_trade_each_cycle: true`): esteira contínua com mandatory pick após quality gate dual soft + vetoes HARD de microestrutura; redirect inter-símbolo se âncora degradada (Z<-0.50) e par forte (Z>+0.50). Cooldown pós-LOSS, blackout de broker e stubs sniper (Hurst/BB) não bloqueiam.
- **Bloqueio absoluto** somente para falhas técnicas: `data`, `predict_error`, `training`, `deploy_ok=false`, Triton fail-closed e reconciliação pendente.
- **Ranking TCN × Z-Score**: `market_decision_score = tcn × max(0.1, 1+z)` — LightGBM validado ranqueia acima de TCN bruto degradado.
- **Gatilho D-SQUEEZE (`[D-SQUEEZE]`)**: quando `predicted_payoff_edge < -0.15` em compressão micro (`bb_width < 0.06` ou `micro_tick_acceleration < 0`), o resolver rebaixa `trade_score` para **0.52**, comprimindo stake — sem inverter a direção da TCN. `bb_width_adaptive_squeeze` está **desabilitado** nos settings atuais.
- **Recovery**: rotação de símbolo após loss linear; soft D'Alembert paramétrico; reset de risco somente quando `pending_loss` zera.
- **Loss protection**: `min_direction_margin: 0.03`; caps edge/Z 999; quality guard em modo mandatorio nunca suspende o cluster por soft alone.
- **Settlement**: janela de tolerância **90 s** com reconciliação passiva (portfolio + Redis); sem reinício forçado por timeout pós-liquidação.
- **Starvation**: após **6** quality skips, pisos de margem/edge/Z decaem; Convicção Progressiva (−20%/5 skips em recovery).
- **Reconexão**: `release_trading_cycle_after_reconnect` invalida assinatura/epoch e reduz warm-up micro quando há `pending_loss`; log `RECOV: ciclo liberado`.
- **Assinatura**: gravada somente após cluster executado; quality skip não consome o candle. Prefixos legados `m5`/`m15` mapeiam 120/600 s.
- **Watchdog de ingestão**: em modo contínuo, reconecta WebSocket se ticks pararem por >**25 s** (`watchdog_stale_tick_seconds`), persistindo snapshot de risco antes.

---

## Observabilidade

Logs em `logs/engine.log` (formato `AetherFormatter`):

- `CFG decisao` — modo DL, lookback, histórico de treino, execução obrigatória
- `FASE TREINO` / `FASE OPERACAO` — transição entre fases
- `DL` / `DL REC` — linha curta: `exec`, `bias` (ajuste direcional), `skip` (bloqueio técnico)
- `EXEC_SEL` — símbolo escolhido com `ord=`, `dl=`, métricas `s`/`v`/`r`, indicadores e alternativas
- `D'ALEMBERT`, `RISK: RECOVERY`, `RISK: WIN operacional`, `KELLY: consensus retention` — sizing soft recovery e penalidade de consenso (quando habilitada)
- `SESSAO INICIADA | Alvo de 2,60%: $XX.XX | Stop Loss: DESATIVADO` — bootstrap de meta por sessão ativa
- `TRITON_TIMEOUT_FALLBACK`, `WATCHDOG: STALE_DATA` — resiliência de inferência e ingestão
- `[AETHER] STOP_WIN` — meta atingida; purge Redis; encerramento CRITICAL
- `meta_payoff_edge_zscore` / `edge_zscore` — Z-Score estatístico do edge LightGBM no ranking
- `[AETHER] EXECUTION_FLOW` — telemetria de fluxo mandatário contínuo (substitui semântica legada `QUALITY_GUARD`)
- `CICLO: cooling-down` — único log de cooldown pós-LOSS no agendamento; silêncio absoluto durante o timer
- `DATA_SIG: cache invalidado` — assinatura micro+macro mudou; inferência reinicializada
- `[API_GUARD]` — telemetria reativa de manutenção do broker (sem bloqueio de ciclo em modo mandatário)
- `[D-SQUEEZE]` — downgrade de score em compressão micro (`bb_width`, `tick_accel`, `predicted_payoff_edge`, `score`)
- `Loop reinicializado de forma transparente` — recovery pós-deadlock sem `sys.exit`; persistência de emergência antes do reset de contadores
- `REGIME_GUARD` — telemetria do persistence guard (`FREEZE`, skip; flip **não** aplicado em produção)
- Linha `IND` — `MARGIN`, `NEUTRAL`, `META_VETO` (`none`/`soft`/`hard`)
- `SETTLE:` — enfileiramento/consumo da fila Redis de liquidação por instabilidade do broker

Mensagens repetidas são deduplicadas (`log_dedupe`, `CooldownDeduplicationFilter`). Cada ciclo e bloco de treino são separados por linha em branco.

Monitor opcional: `python app/scripts/monitor/live_monitor.py`

---

## Stack e qualidade

- **Python 3.13.12**, `asyncio` (`app/aether_asyncio.py` — wrapper `asyncio.run` que silencia ruído de debug), NumPy, Polars, PyTorch (TCN / LSTM / GRU)
- **Deriv** PAT + REST OTP + WebSocket (`api_config` em settings; ver `docs/deriv-api.md`)
- **Infra**: Redis, TimescaleDB, MinIO, NVIDIA Triton (gRPC, timeout **0,50 s**), meta-regressor LightGBM (HTTP 8005)
- **CI / pre-commit**: Ruff, Interrogate, Vulture, limite 300 linhas/arquivo, pytest com **100%** de cobertura em `app/src` (**287** arquivos de teste; **~226** módulos em `app/src`)

Requisito local: ambiente Conda **`deriv-api`** (Python 3.13.12). Configuração em [`config/python.json`](config/python.json).

Windows: abra **Anaconda PowerShell Prompt**, `conda activate deriv-api`:

```powershell
conda activate deriv-api
pip install -r app/requirements.txt -r app/requirements-dev.txt
python app/scripts/operations/deriv_pat_connect.py
python train.py
python run.py
```

WSL:

```bash
cd /mnt/c/Users/<seu-usuario>/Desktop/aether-quantum-engine
make app-setup-wsl
source ~/.bashrc
make app-install
make app-test
make app-lint
```

Pre-commit (na raiz do repo):

```powershell
conda activate deriv-api
python -m pre_commit run --all-files
```

WSL: `make app-pre-commit-run`

---

## Execução ao vivo

1. Configure `.env` com PAT e App ID (app PAT em developers.deriv.com).
2. `conda activate deriv-api` e instale dependências.
3. Suba a infra: `make docker-up` (quando `infra.enabled: true`).
4. Treine os modelos: `python train.py` ou `app/scripts/batch/launch-train.bat`.
5. Valide checkpoints DL em `data/dl/`.
6. Execute o motor: `python run.py`, `make run` ou `launch-all-demo.bat` / `launch-all-live.bat`.

O motor exige `deep_learning.enabled: true` e checkpoints válidos em `data/dl/`. Treino e execução são processos separados — `train.py` grava os modelos; `run.py` só opera.

---

## Referências

- [docs/arquitetura.md](docs/arquitetura.md) — fluxos técnicos
- [docs/medallion.md](docs/medallion.md) — filosofia quant e parâmetros de qualidade
- [docs/deriv-api.md](docs/deriv-api.md) — API Deriv
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — histórico de releases
