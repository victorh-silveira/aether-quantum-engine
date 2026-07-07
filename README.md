# Aether Quantum Engine 2.0

[![Python](https://img.shields.io/badge/Python-3.13.12-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](app/tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono para a Deriv: decisão por **Deep Learning** (TCN/LSTM/GRU) nos índices **Drift** (`RDBEAR`, `RDBULL`), contratos **RISE_FALL** de **60 s (M1)** com contexto macro **M15 (900 s)**, meta-regressor LightGBM de expectativa de retorno contínuo, e recuperação **Martingale Geométrico** (`Effective_Base × 2^n`) quando há passivo pendente.

A operação divide-se em duas fases: **FASE TREINO** (nenhuma ordem até todos os modelos concluírem o treino da sessão) e **FASE OPERACAO** (seletiva por padrão ou **modo contínuo** com `mandatory_trade_each_cycle: true` — uma ordem por ciclo, sem SKIP de qualidade).

Documentação: [arquitetura](docs/arquitetura.md) | [metodologia quant](docs/medallion.md) | [estrutura do repo](docs/structure.md) | [infra Docker](docs/infra-docker.md) | [Deriv API](docs/deriv-api.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` + `TickBuffer` + `AetherWatchdog` | WebSocket Deriv dual-timeframe: OHLC macro M15 (900 s) para DL/regimes + OHLC micro M1 (60 s) para gatilho do ciclo; ticks agregados por barra fechada; watchdog reconecta stream em inanição (>30 s) |
| Fases | `_training_phase_gate` | Suspende a operação até todos os modelos concluírem o treino da sessão |
| Predição DL | `decision_bridge` + `dl_predict_build` + TCN/LSTM/GRU | **34 features** TCN; bundle cross-symbol 39D antes do prefetch meta; inferência Triton gRPC com timeout 2 s e fallback TorchScript |
| Meta GBDT | `meta_classifier_client` + `aether-meta-classifier` | Regressão tabular **39D** (`LGBMRegressor` huber); retorna `predicted_payoff_edge` contínuo |
| Direção | `execution_direction_resolver` + `meta_payoff_regression` | TCN define macro; edge `> 0` preserva score orgânico; edge `< -0.15` em squeeze rebaixa para **0.52** (`[D-SQUEEZE]`); `direction_margin = abs(P(lado) − 0.50)` |
| Qualidade | `execution_quality_gate` | Janelas dinâmicas: margem mín. **0.06** (regular) / **0.12** (recovery); payoff meta mín. **0.01** / **0.04**; suspende cluster com `[AETHER] QUALITY_GUARD` |
| Execução | `ExecutionManager` + `execution_collect` | Ranking por `market_decision_score`; mandatory pick quando configurado |
| Risco | `RiskManager` + `dlambert_sizing` + `consensus_stake_penalty` | Kelly + Martingale `U × 2^n` em recovery; bypass de consenso com `pending_total > 0` |
| Resiliência | `graceful_shutdown` + `watchdog_service` + `post_settlement_cycle` + `api_maintenance_guard` | Fast-path stop win; cancelamento de fila Redis/settlement; hibernação cooperativa em manutenção do broker; teto 2× incompleto → `sys.exit(0)` |
| Concorrência | `StateManager` + `orchestrator_atomic_state` + `session_persistence_barrier` | `asyncio.Lock` central serializa inferência DL, liquidação e persistência; leituras de infra via `read_cached_balance` sem bloquear o lock |
| Cache M1+M15 | `orchestrator_data_signature` | Assinatura multi-timeframe invalida inferência redundante na mesma fronteira de minuto |
| Estado | `StateManager` + `redis_state_pipeline` + `orchestrator_persistence` | Snapshot atômico MULTI/EXEC; persistência locked/unlocked; barreira pós-reset linear D'Alembert |
| Inferência | `TritonGrpcClient` | Canal `grpc.aio.insecure_channel` persistente; timeout 2 s; predições paralelas via `asyncio.gather`; fallback local em timeout |
| Mercado TS | `TimescaleMarketWriter` | Ticks e barras OHLC macro M15 (900 s) e micro M1 (60 s) para backtest |
| Modelos | `MinioModelStore` + cache `data/dl/` | Checkpoints DL como source of truth remoto; sanity estressado no startup |

Ciclo do orquestrador: `orchestrator.cycle_interval_seconds` (**60 s / M1**). Contexto DL: `data_handler.granularity` (**900 s / M15**, tensor `[1, 48, 34]`). Contrato: `risk_management.params.duration` (**60 s**, RISE_FALL alinhado ao fechamento M1).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (``RDBEAR`, `RDBULL`; âncora `RDBULL`) |
| `data_handler` | `granularity` (macro M15), `micro_granularity` (M1), `history_bars`, `fetch_count`, `buffer_limit` |
| `deep_learning` | `arch`, `lookback`, `calibration`, thresholds, `deploy_gate` |
| `orchestrator.execution` | `direction_scoring`, `dynamic_threshold`, `quality_gate`, `mandatory_trade_each_cycle`, mean-reversion, settlement |
| `orchestrator` | `watchdog_*`, `cycle_interval_seconds`, execução, settlement |
| `risk_management.kelly` | Kelly, martingale, `consensus_penalty_*`, `penalty_smoothing_*`, `martingale_hard_cap_bankroll_pct` |
| `risk_management.params` | `duration: 60`, stakes, `compounding_enabled`, `compounding_rate_daily`, `session_start_balance` (opcional) |
| `trading` | `demo` / `live` |
| `infra` | Redis, TimescaleDB, MinIO, Triton (`enabled`, `fail_fast`, `grpc_url`), meta-regressor (`enabled`, `url`) |

## Ambiente híbrido Docker

O motor (`run.py` / `train.py`) roda no host Conda/WSL. Redis, TimescaleDB, MinIO, **Triton** e **meta-regressor** sobem via Docker em `localhost`:

```bash
make docker-up
```

O target aplica `host-prereq.sh` (`vm.overcommit_memory=1` no WSL) e sobe Redis com AOF `everysec`. Validação:

```bash
docker exec -it aether-redis redis-cli CONFIG GET appendonly
docker exec -it aether-redis redis-cli CONFIG GET appendfsync
```

Com `infra.enabled: true`, o startup valida os serviços (fail-fast), sincroniza TorchScript no Triton e executa **sanity estressado** (RSI/CMO/vol extremos) antes do WebSocket Deriv. Detalhes em [docs/infra-docker.md](docs/infra-docker.md).

Variáveis na raiz (`.env` único — Deriv + infra Docker):

| Variável | Uso |
|----------|-----|
| `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` | Conta Deriv |
| `AETHER_PG_USER`, `AETHER_PG_PASSWORD`, `AETHER_PG_DB` | TimescaleDB |
| `AETHER_MINIO_ACCESS_KEY`, `AETHER_MINIO_SECRET_KEY` | MinIO |
| `AETHER_TRITON_GRPC`, `AETHER_TRITON_HTTP` | Triton Inference Server |
| `AETHER_META_CLASSIFIER_URL` | Meta-classificador LightGBM (padrão `http://localhost:8005`) |

Copie `cp .env.example .env` e preencha o PAT. Validação Deriv: `python app/scripts/operations/deriv_pat_connect.py`.

---

## Gerenciamento de risco

- **Kelly fracionário** com win rate dinâmico e tetos `max_stake_pct`.
- **Consensus Entropy Penalty**: atenuação convexa de `f*` quando a ordem diverge da maioria dos votos técnicos (`call_votes`/`put_votes`), ponderando `di_diff`, `cmo` e `rsi`; em baixo consenso, stake reduzida ao piso mínimo da API.
- **Penalty smoothing em recovery**: com drawdown pendente ou `consecutive_losses > 0` e `trade_score > 0.70`, a penalidade convexa é suavizada em 40% (`penalty_smoothing_factor`), permitindo stakes maiores para recuperação sem violar o CAP.
- **Recovery financeiro persistente**: WIN operacional **não** zera `consecutive_losses` enquanto `pending_loss > 0`; o motor permanece em modo Martingale até o drawdown pendente ser extinto por retornos reais.
- **CAP martingale 4%**: `martingale_hard_cap_bankroll_pct: 0.04` limita stake máxima sobre a banca.
- **Stop win por sessão ativa**: no boot, captura o saldo vivo (Deriv) ou override `session_start_balance`; meta = `banca_inicial × compounding_rate_daily` (padrão **2,60%**). Quando `pnl_sessao >= meta`, dispara **fast-path** (`clear_current_session_redis_keys` → cancelamento da fila de settlement → `graceful_shutdown(fast_path=True)`).
- **Stop loss desativado**: não há disjuntor de perda diária; o Martingale opera sem teto de drawdown imposto pelo motor.
- **Martingale de recovery** quando há perda pendente: stake cobre perda integral + alvo derivado do payout, com fatiamento progressivo em sequências longas.
- Cooldown por símbolo após sequência de losses (`symbol_loss_cooldown_candles`).

---

## Fases, recovery e execução

- **FASE TREINO**: ao iniciar a sessão, todo símbolo retreina pelo menos uma vez. Enquanto qualquer modelo não concluir, nenhuma ordem é enviada.
- **FASE OPERACAO seletiva** (`mandatory_trade_each_cycle: false`): opera quando o melhor candidato passa no gate de qualidade (margem direcional e payoff meta acima dos pisos do regime).
- **FASE OPERACAO contínua** (`mandatory_trade_each_cycle: true`): uma ordem por ciclo quando há candidato válido; sinais com `direction_margin` insuficiente são rejeitados; em recovery o fallback obrigatório não contorna o veto coletivo do quality gate.
- **Gatilho D-SQUEEZE (`[D-SQUEEZE]`)**: quando `predicted_payoff_edge < -0.15` em compressão M1 (`bb_width < 0.06` ou `micro_tick_acceleration < 0`), o resolver rebaixa `trade_score` para **0.52**, comprimindo stake via consensus penalty até o piso de $1.00 da Deriv — sem inverter a direção da TCN.
- **Trava Hurst em recovery N2+**: com `consecutive_losses >= 2`, piso de score elevado logaritmicamente; `recovery_skip_counter` no Redis decai o limiar Hurst.
- **Bloqueio absoluto** somente para falhas técnicas: `data`, `predict_error`, `training`, `deploy_ok=false`.
- **Recovery**: ranking de mercado com diversificação de símbolo; martingale com convicção mínima 0.64 e `val_accuracy` ≥ 0.62; reset de risco somente quando `pending_loss` zera.
- **Watchdog de ingestão**: em modo contínuo, reconecta WebSocket se ticks pararem por >30 s (`watchdog_stale_tick_seconds`), persistindo snapshot de risco antes.

---

## Observabilidade

Logs em `logs/engine.log` (formato `AetherFormatter`):

- `CFG decisao` — modo DL, lookback, histórico de treino, execução obrigatória
- `FASE TREINO` / `FASE OPERACAO` — transição entre fases
- `DL` / `DL REC` — linha curta: `exec`, `bias` (ajuste direcional), `skip` (bloqueio técnico)
- `EXEC_SEL` — símbolo escolhido com `ord=`, `dl=`, métricas `s`/`v`/`r`, indicadores e alternativas
- `SKIP` — ciclo pulado por bloqueio técnico ou recovery sem Hurst persistente (modo seletivo)
- `MARTINGALE`, `RISK: RECOVERY`, `RISK: WIN operacional`, `KELLY: consensus retention` — sizing, recovery financeiro e penalidade de consenso
- `SESSAO INICIADA | Alvo de 2,60%: $XX.XX | Stop Loss: DESATIVADO` — bootstrap de meta por sessão ativa
- `TRITON_TIMEOUT_FALLBACK`, `WATCHDOG: STALE_DATA` — resiliência de inferência e ingestão
- `[AETHER] QUALITY_GUARD` — ciclo descartado por margem TCN ou payoff meta insuficiente (`linear`, `pending_loss`)
- `DATA_SIG: cache invalidado` — assinatura M1+M15 mudou; inferência reinicializada
- `[API_GUARD]` — hibernação cooperativa durante manutenção ou reset de liquidez do broker
- `[D-SQUEEZE]` — downgrade de score em compressão M1 (`bb_width`, `tick_accel`, `predicted_payoff_edge`, `score`)
- `CICLO: ciclo pos-liquidacao incompleto` — retry pós-liquidação; após 2 falhas consecutivas, persistência de emergência e encerramento atômico
- Liquidação e resumo de cluster após settlement

Mensagens repetidas são deduplicadas (`log_dedupe`). Cada ciclo e bloco de treino são separados por linha em branco.

Monitor opcional: `python app/scripts/monitor/live_monitor.py`

---

## Stack e qualidade

- **Python 3.13.12**, `asyncio`, NumPy, Polars, PyTorch (TCN / LSTM / GRU)
- **Deriv** PAT + REST OTP + WebSocket (`api_config` em settings; ver `docs/deriv-api.md`)
- **Infra**: Redis, TimescaleDB, MinIO, NVIDIA Triton (gRPC), meta-regressor LightGBM (HTTP 8005)
- **CI / pre-commit**: Ruff, Interrogate, Vulture, limite 300 linhas/arquivo, pytest com **100%** de cobertura em `app/src` (~1893 testes)

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
make setup-wsl
source ~/.bashrc
make install
make test
make lint
```

Pre-commit (na raiz do repo):

```powershell
conda activate deriv-api
python -m pre_commit run --all-files
```

WSL: `make pre-commit-run`

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
