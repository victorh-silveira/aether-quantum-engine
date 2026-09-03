# Aether Quantum Engine

[![Python](https://img.shields.io/badge/Python-3.13.12-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Coverage](https://img.shields.io/badge/Coverage-100%25%20gate-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Architecture](https://img.shields.io/badge/Architecture-Clean%20%7C%20Domain--pure-1F4E79)](docs/arquitetura.md)
[![Asyncio](https://img.shields.io/badge/Runtime-asyncio%20%7C%20WSL-00599C?logo=python&logoColor=white)](AGENTS.md)
[![QA](https://img.shields.io/badge/QA-ruff%20%7C%20pre--commit%20%7C%20bandit-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![PyTorch](https://img.shields.io/badge/DL-PyTorch%20TCN%20M5-EE4C2C?logo=pytorch&logoColor=white)](docs/engineering-deep-learning.md)
[![Polars](https://img.shields.io/badge/DataFrame-Polars%20SSOT-CD792C)](docs/engineering-python-deps.md)
[![Quant](https://img.shields.io/badge/Risk-Kelly%20Single--Strike%201%25-0B3D91)](docs/llm-trading-doctrine.md)
[![Infra](https://img.shields.io/badge/Infra-Redis%20%7C%20Timescale%20%7C%20MinIO-DC382D?logo=redis&logoColor=white)](docs/infra-docker.md)
[![Deriv](https://img.shields.io/badge/Market-Deriv%201HZ75V%20%7C%20RISE__FALL%20M5-111111)](docs/deriv-api-aether.md)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/victorh-silveira/aether-quantum-engine?display_name=tag&label=Release)](https://github.com/victorh-silveira/aether-quantum-engine/releases)

Motor quantitativo assíncrono para a Deriv: decisão por **Deep Learning** (TCN/LSTM/GRU) no índice sintético de volatilidade **Volatility 75 (1s)** (`1HZ75V`), contratos **RISE_FALL** de **5 m (M5)** com label TCN em **N=1 vela M5** (`triple_barrier`), lookback **30**, micro/MINI **300 s** (500 velas) e contexto macro D1 **86400 s** (365 barras diárias de treino), meta-regressor LightGBM (**43D**) de expectativa de retorno contínuo (single-symbol), e **sizing Kelly Single-Strike** (alvo de 4.31% da banca em tacada única M5; Soft Recovery amort **2/3** em RECOVER). Sem Triton: inferência eager/CUDA local; Docker profiles `core,ml`.

A operação divide-se em duas fases: **FASE TREINO** (nenhuma ordem até checkpoint/sessão prontos; `online_training` **false** no DEMO) e **FASE OPERACAO** continua (`mandatory_trade_each_cycle: false`, `force_trade_every_cycle: false`, `invert_exec_side: false`): o ciclo avalia candidato a cada **300 s** (fechamento da barra M5) via TCN + fusao EV + microestrutura balanceada M5 + signal_skip 1.1. Meta é **opcional** para execução (`require_meta_for_execution: false`); inferência TCN = eager/CUDA local no host.

Documentação: [AGENTS.md](AGENTS.md) (agentes) | [matriz de cobertura](docs/agent-coverage.md) | [arquitetura](docs/arquitetura.md) | [estrutura e módulos](docs/structure.md) | [metodologia quant](docs/medallion.md) | [infra Docker](docs/infra-docker.md) | [Deriv API](docs/deriv-api.md) | [Deriv para agentes](docs/deriv-api-aether.md) | [índice docs](docs/README.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` + `TickBuffer` + `AetherWatchdog` | WebSocket Deriv dual-timeframe: OHLC macro **86400 s** (D1) para DL/regimes + OHLC micro **300 s** (M5) para gatilho do ciclo; ticks agregados por barra fechada; watchdog reconecta stream em inanição (`watchdog_stale_tick_seconds` **300**) |
| Fases | `_training_phase_gate` | Suspende a operação até todos os modelos concluírem o treino da sessão |
| Predição DL | `decision_bridge` + `dl_predict_*` + TCN | **34 features** TCN; bundle meta **43D**; inferência eager/CUDA local |
| Meta GBDT | `meta_classifier_client` + `aether-meta-classifier` | Regressão tabular **43D**; `predicted_payoff_edge` contínuo (opcional para execução) |
| Z-Score payoff | `payoff_edge_zscore` | Janela adaptativa 15–45; `meta_payoff_edge_zscore` |
| Direção | `execution_direction_*` (resolver + checks + persistence + meta_edge + discordance) | TCN define lado (thresholds **0.46/0.34**); zona neutra **off**; anti-loss microestrutura M5; SIDE_EQ antecipado |
| Rotulagem DL | `dl_labels` + `LabelSpec` | SSOT `triple_barrier` (horizonte N=1 vela M5) |
| Quality / starvation | `execution_quality_gate*` | Dual soft TCN+meta; pisos regulares de margem/ADX **0.0**; starvation a partir de **6** skips; edge decay a partir de **8** |
| Ranking | `execution_market_rank` | Score `tcn × max(0.1, 1+z)` |
| Execução | `ExecutionManager` + lotes fracionados | Proposta atômica; RISE_FALL **5 m** (ops fixo M5) |
| Risco | `RiskManager` + Kelly Single-Strike / Soft Recovery | Kelly Single-Strike 4.31% (alvo de 4.31% da banca em payout 0.85); Soft Recovery amort **2/3** em RECOVER (`cover_multiple` **1.10**, `max_safe_stake_pct`) |
| Concorrência | `StateManager` + barreira atômica | Lock serializa inferência, liquidação e persistência |
| Inferência | PyTorch eager / CUDA | Checkpoint local `data/dl/`; motor no host |

Ciclo do orquestrador: `orchestrator.cycle_interval_seconds` (**120 s**) / `signature_boundary_seconds` (**300 s**) / `exec_empty_retry_seconds` (**120 s**). Contexto DL: `data_handler.granularity` (**86400 s**), micro/MINI **300 s**, tensor `[1, 30, 34]` (`deep_learning.lookback` **30**). Contrato: `risk_management.params.duration` (**5** m); label `deep_learning.label_horizon_bars` (**1** vela M5).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (`1HZ75V`; ancora `1HZ75V`) |
| `data_handler` | `granularity` (macro **86400 s**), `micro_granularity` / `mini_granularity` (**300 s**), historico treino **365** barras D1 |
| `deep_learning` | `arch`, `lookback` (**30**), `online_training` **false**, calibration, thresholds **0.46/0.34**, `deploy_gate` |
| `orchestrator.execution` | `mandatory_trade_each_cycle: false`, `force_trade_every_cycle: false`, `invert_exec_side: false`, `scale_vision.fusion_*`, `signal_skip` 1.1, settlement **600 s** |
| `risk_management.kelly` | Stake Kelly Single-Strike 4.31% (`fraction: 0.08`, stop-win Kelly **4.31%**, tetos stop-win ate **5%**) |
| `risk_management.soft_recovery` | RECOVER: amort **2/3**, cover **1.10**, linear3 **2.5%** |
| `orchestrator.execution.side_equilibrium` | Leis dos pequenos/grandes números CALL/PUT (small-N hard skip; large-N soft Kelly) |
| `infra` | Redis, Timescale, MinIO, meta-classifier, loss-classifier |

## Ambiente híbrido Docker

O motor (`run.py` / `train.py`) roda no host Conda/WSL. Redis, TimescaleDB, MinIO, **meta-regressor** e **loss-classifier** sobem via Docker em `localhost`:

```bash
make docker-up
```

Pipeline: `host-prereq` → `compose up` (profiles `DOCKER_PROFILES`, padrão `core,ml`) → wait healthy → `timescale-lifecycle` → `docker-hydrate` → `docker-smoke`. Redis sobe com AOF `everysec`.

| Target | Uso |
|--------|-----|
| `make docker-up` | Stack completa (core+ml) |
| `make docker-up-core` | Só Redis, Timescale e MinIO |
| `make docker-rebuild` | Rebuild meta/loss e recarrega pkls (preserva TCN) |
| `make docker-smoke` | Valida endpoints da stack |

Detalhes em [docs/infra-docker.md](docs/infra-docker.md).

```bash
docker exec -it aether-redis redis-cli CONFIG GET appendonly
docker exec -it aether-redis redis-cli CONFIG GET appendfsync
```

Com `infra.enabled: true`, o startup valida os serviços (fail-fast) e carrega o checkpoint TCN local antes do WebSocket Deriv.

Variáveis na raiz (`.env` único — Deriv + infra Docker):

| Variável | Uso |
|----------|-----|
| `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` | Conta Deriv |
| `AETHER_PG_USER`, `AETHER_PG_PASSWORD`, `AETHER_PG_DB` | TimescaleDB |
| `AETHER_MINIO_ACCESS_KEY`, `AETHER_MINIO_SECRET_KEY` | MinIO |
| `AETHER_META_CLASSIFIER_HTTP` | Meta-classificador LightGBM (padrão `http://localhost:8005`) |
| `AETHER_DOCKER_HEALTH_TIMEOUT` | Timeout do wait healthy (padrão `300`) |

Copie `cp .env.example .env` e preencha o PAT. Validação Deriv: `python app/scripts/operations/deriv_pat_connect.py`.

---

## Gerenciamento de risco

- **Kelly + Soft Recovery** (`risk_stake_calc` + `soft_recovery_policy`):
  - **EXPLORE** (`pending_loss == 0` e `consecutive_losses_linear == 0`): stake = Kelly fracionário (`fraction: 0.08`, compressão 40%, teto **3,5%** da banca) — tag `EXPLORE_KELLY`.
  - **RECOVER** (`pending_loss > 0` ou `linear >= 1`): Soft Recovery cover equilibrado (`cover_multiple` **1.10**, amort **2/3**), teto `max_safe_stake_pct` — tag `RECOVER_DAL_Ln`.
- Soft Recovery cobre `pending/payout * cover_multiple` em **2–3** ciclos (amort **2/3**) com teto `max_safe_stake_pct` / linear3 **3.5%**.
- **Consensus Entropy Penalty**: presente no código; nos settings atuais `consensus_penalty_enabled: false`.
- **Side equilibrium (LLN)**: `side_equilibrium` — small-N (janela 12, `n_min=2`) hard skip se WR baixo ou frequência enviesada; large-N (janela 100, `n_min=40`) soft penalty (`kelly_mult_soft` → escala f*). Com contagens 0/0 o gate faz `pass` (amostra insuficiente) — esperado no início da sessão.
- **Recovery financeiro persistente**: WIN operacional **não** zera `consecutive_losses` enquanto `pending_loss > 0`; reconciliação de stake downgrade preserva drawdown real.
- **Stop win por sessão ativa**: meta = `banca_inicial × 3,00%` (banca ≥ $100) ou **$10** fixo (banca &lt; $100); `finalize_stop_win_shutdown` purge Redis + log CRITICAL + fast-path.
- **Stop loss desativado**: sem disjuntor de perda diária interno.
- **Lotes fracionados**: stakes acima de `max_single_stake_limit` (padrão $200) divididas em N ordens com proposta atômica por sub-lote; falha técnica de proposta aborta o cluster sem inflar `pending_loss`.
- Cooldown por símbolo após sequência de losses (`symbol_loss_rotation_cycles`): com universo single-symbol (`R_10`) o default operacional e `0` para nao esvaziar o unico ativo.
- **Proteção contra loss** (`execution_loss_protection`): caps edge/Z **999**; `min_direction_margin` operacional **0.0** nos settings atuais.
- **Starvation / edge**: após **6** quality skips os pisos decaem; o piso de `predicted_payoff_edge` relaxa a partir de **8** skips até `edge_decay_floor: 0.0`, com recovery relax até `-0.55`.

---

## Fases, recovery e execução

- **FASE TREINO**: ao iniciar a sessão, todo símbolo retreina pelo menos uma vez. Enquanto qualquer modelo não concluir, nenhuma ordem é enviada.
- **FASE OPERACAO** continua (`mandatory_trade_each_cycle: false`, `force_trade_every_cycle: false`): a cada fronteira de **60 s** o motor avalia candidato via TCN + fusao EV + signal_skip 1.1 (quality gate amplo **fora**).
- **Bloqueio absoluto** somente para falhas técnicas: `data`, `predict_error`, `training`, `deploy_ok=false`, e reconciliação pendente.
- **Ranking TCN × Z-Score**: `market_decision_score = tcn × max(0.1, 1+z)` — LightGBM validado ranqueia acima de TCN bruto degradado.
- **Gatilho D-SQUEEZE (`[D-SQUEEZE]`)**: quando `predicted_payoff_edge < -0.15` em compressão micro (`bb_width < 0.06` ou `micro_tick_acceleration < 0`), o resolver rebaixa `trade_score` para **0.52**, comprimindo stake — sem inverter a direção da TCN. `bb_width_adaptive_squeeze` está **desabilitado** nos settings atuais.
- **Recovery**: Soft Recovery cover equilibrado (amort **2/3**, `cover_multiple` **1.10**) após loss linear; reset de risco somente quando `pending_loss` zera.
- **Loss protection**: caps edge/Z 999; quality guard em modo mandatório prioriza esteira contínua (soft alone não congela o cluster).
- **Settlement**: janela de tolerância **600 s** com reconciliação passiva (portfolio + Redis); pós-EXEC_EMPTY em recovery alinha a próxima fronteira (cap de retry).
- **Starvation**: após **6** quality skips, pisos decaem; edge meta relaxa a partir de **8** skips; Convicção Progressiva (−20%/5 skips em recovery).
- **Persistence / SIDE_EQ**: após 2 losses no mesmo lado tenta flip (toxic escape, edge positivo preservado); SIDE_EQ hard-skip tenta o oposto e pode marcar `side_eq_escape_edge_kept`.
- **Reconexão**: `release_trading_cycle_after_reconnect` invalida assinatura/epoch e reduz warm-up micro quando há `pending_loss`; log `RECOV: ciclo liberado`.
- **Assinatura**: gravada somente após cluster executado; quality skip não consome o candle. Prefixos legados `m5`/`m15` mapeiam 60/7200 s.
- **Watchdog de ingestão**: em modo contínuo, reconecta WebSocket se ticks pararem por >**300 s** (`watchdog_stale_tick_seconds`), persistindo snapshot de risco antes.
- **Deriv API**: REST/PAT com retry em 502/503/504 e respeito a `retry_after` Cloudflare.

---

## Observabilidade

Logs em `logs/engine.log` (formato `AetherFormatter`):

- `CFG decisao` — modo DL, lookback, histórico de treino, execução obrigatória
- `FASE TREINO` / `FASE OPERACAO` — transição entre fases
- `DL` / `DL REC` — linha curta: `exec`, `bias` (ajuste direcional), `skip` (bloqueio técnico)
- `EXEC_SEL` — símbolo escolhido com `ord=`, `dl=`, métricas `s`/`v`/`r`, indicadores e alternativas
- `EXPLORE_KELLY`, `RECOVER_DAL_Ln`, `RISK: RECOVERY`, `RISK: WIN operacional` — sizing Kelly/Soft e estado de recovery
- `SIDE_EQ` — equilíbrio CALL/PUT (leis dos pequenos/grandes números)
- `SESSAO INICIADA | Alvo de 3,00%: $XX.XX | Stop Loss: DESATIVADO` — bootstrap de meta por sessão ativa
- `WATCHDOG: STALE_DATA` — resiliência de ingestão
- `[AETHER] STOP_WIN` — meta atingida; purge Redis; encerramento CRITICAL
- `meta_payoff_edge_zscore` / `edge_zscore` — Z-Score estatístico do edge LightGBM no ranking
- `[AETHER] EXECUTION_FLOW` — telemetria de fluxo mandatário contínuo (substitui semântica legada `QUALITY_GUARD`)
- `CICLO: cooling-down` — único log de cooldown pós-LOSS no agendamento; silêncio absoluto durante o timer
- `DATA_SIG: cache invalidado` — assinatura micro+macro mudou; inferência reinicializada
- `[API_GUARD]` — telemetria reativa de manutenção do broker (sem bloqueio de ciclo em modo mandatário)
- `[D-SQUEEZE]` — downgrade de score em compressão micro (`bb_width`, `tick_accel`, `predicted_payoff_edge`, `score`)
- `Loop reinicializado de forma transparente` — recovery pós-deadlock sem `sys.exit`; persistência de emergência antes do reset de contadores
- `REGIME_GUARD` — telemetria do persistence guard (`FREEZE`, skip ou flip toxic escape)
- Linha `IND` — `MARGIN`, `NEUTRAL`, `META_VETO` (`none`/`soft`/`hard`)
- `SETTLE:` — enfileiramento/consumo da fila Redis de liquidação por instabilidade do broker

Mensagens repetidas são deduplicadas (`log_dedupe`, `CooldownDeduplicationFilter`). Cada ciclo e bloco de treino são separados por linha em branco.

Monitor opcional: `python app/scripts/monitor/live_monitor.py`

---

## Stack e qualidade

- **Python 3.13.12**, `asyncio` (`app/aether_asyncio.py` — wrapper `asyncio.run` que silencia ruído de debug), NumPy, Polars, PyTorch (TCN / LSTM / GRU)
- **Deriv** PAT + REST OTP + WebSocket (`api_config` em settings; ver `docs/deriv-api.md`)
- **Infra**: Redis, TimescaleDB, MinIO, meta-regressor LightGBM (HTTP 8005), loss-classifier (HTTP 8006); TCN eager/CUDA local
- **CI / pre-commit**: Ruff, Interrogate, Vulture, limite 300 linhas/arquivo, pytest com **100%** de cobertura em `app/src` (**306** arquivos de teste; **246** módulos em `app/src`)

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
6. Execute o motor: `python run.py`, `make run` ou `launch-all-demo.bat`.

O motor exige `deep_learning.enabled: true` e checkpoints válidos em `data/dl/`. Treino e execução são processos separados — `train.py` grava os modelos; `run.py` só opera.

Fluxo tipico single-symbol: treinar TCN `R_10` (`data/dl/R_10.pth`) → treinar meta single-symbol (`--symbols R_10`). Artefatos Drift legados no disco nao sao apagados automaticamente.

---

## Referências

- [docs/arquitetura.md](docs/arquitetura.md) — fluxos técnicos
- [docs/medallion.md](docs/medallion.md) — filosofia quant e parâmetros de qualidade
- [docs/deriv-api.md](docs/deriv-api.md) — API Deriv
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — histórico de releases
