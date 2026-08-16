# Aether Quantum Engine 2.0

[![Python](https://img.shields.io/badge/Python-3.13.12-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](app/tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono para a Deriv: decisão por **Deep Learning** (TCN/LSTM/GRU) no índice **Volatility 10** (`R_10`), contratos **RISE_FALL** de **5 m (M5)** com label TCN em **N velas M1** (N ∈ {15,20,…,60} eleito no treino; **SSOT atual N=55** / H55; gap intencional vs settle 5 min), micro/MINI **60 s** e contexto macro **7200 s** (ratio **1:120**), meta-regressor LightGBM (**43D**) de expectativa de retorno contínuo (single-symbol), e **sizing Kelly + Soft Recovery** (Kelly em EXPLORE; Soft Recovery cover pleno amort **1/1** em RECOVER). As chaves de assinatura ainda usam prefixos legados `m5`/`m15` para compatibilidade de cache. Sem Triton: inferência eager/CUDA local; Docker profiles `core,ml`.

A operação divide-se em duas fases: **FASE TREINO** (nenhuma ordem até checkpoint/sessão prontos; `online_training` **false** no DEMO) e **FASE OPERACAO** continua (`mandatory_trade_each_cycle: false`, `force_trade_every_cycle: false`, `invert_exec_side: false`): o ciclo avalia candidato a cada **60 s** via TCN + fusao EV + signal_skip 1.1. Meta é **opcional** para execução (`require_meta_for_execution: false`); inferência TCN = eager/CUDA local no host. Mercado Volatility **24/7**.

Documentação: [AGENTS.md](AGENTS.md) (agentes) | [matriz de cobertura](docs/agent-coverage.md) | [arquitetura](docs/arquitetura.md) | [estrutura e módulos](docs/structure.md) | [metodologia quant](docs/medallion.md) | [infra Docker](docs/infra-docker.md) | [Deriv API](docs/deriv-api.md) | [Deriv para agentes](docs/deriv-api-aether.md) | [índice docs](docs/README.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` + `TickBuffer` + `AetherWatchdog` | WebSocket Deriv dual-timeframe: OHLC macro **7200 s** (assinatura legado `m15`) para DL/regimes + OHLC micro **60 s** (assinatura legado `m5`) para gatilho do ciclo; ticks agregados por barra fechada; watchdog reconecta stream em inanição (`watchdog_stale_tick_seconds` **300**) |
| Fases | `_training_phase_gate` | Suspende a operação até todos os modelos concluírem o treino da sessão |
| Predição DL | `decision_bridge` + `dl_predict_*` + TCN | **34 features** TCN; bundle meta **43D**; inferência eager/CUDA local |
| Meta GBDT | `meta_classifier_client` + `aether-meta-classifier` | Regressão tabular **43D**; `predicted_payoff_edge` contínuo (opcional para execução) |
| Z-Score payoff | `payoff_edge_zscore` | Janela adaptativa 15–45; `meta_payoff_edge_zscore` |
| Direção | `execution_direction_*` (resolver + checks + persistence + meta_edge + discordance) | TCN define lado (thresholds **0.51/0.49**); zona neutra **off**; persistence pode **flipar** (toxic escape) ou skip; SIDE_EQ antecipado; D-SQUEEZE rebaixa score |
| Rotulagem DL | `dl_labels` + `LabelSpec` | SSOT `ma_trend`; `spot_forward` / Triple Barrier via config |
| Quality / starvation | `execution_quality_gate*` | Dual soft TCN+meta; pisos regulares de margem/ADX **0.0**; starvation a partir de **6** skips; edge decay a partir de **8** (`edge_decay_floor` → 0.0) |
| Ranking | `execution_market_rank` | Score `tcn × max(0.1, 1+z)` |
| Execução | `ExecutionManager` + lotes fracionados | Proposta atômica; RISE_FALL **5 m** (ops fixo; label H55) |
| Risco | `RiskManager` + Kelly / Soft Recovery | `EXPLORE_KELLY` (fraction 0.08, teto 3,5%); Soft Recovery cover pleno amort **1/1** em RECOVER (`cover_multiple` **1.50**, `max_safe_stake_pct`) |
| Concorrência | `StateManager` + barreira atômica | Lock serializa inferência, liquidação e persistência |
| Inferência | PyTorch eager / CUDA | Checkpoint local `data/dl/`; motor no host |

Ciclo do orquestrador: `orchestrator.cycle_interval_seconds` / `signature_boundary_seconds` / `exec_empty_retry_seconds` (**60 s**). Contexto DL: `data_handler.granularity` (**7200 s**), micro/MINI **60 s**, tensor `[1, 480, 34]` (`deep_learning.lookback` **480**). Contrato: `risk_management.params.duration` (**5** m, via `horizon_sweep.ops_contract_duration_minutes`); label `deep_learning.label_horizon_bars` (**55** apos ultimo promote). Proporção multi-timeframe **1:120** (60:7200).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (`R_10`; ancora `R_10`) |
| `data_handler` | `granularity` (macro **7200 s**), `micro_granularity` / `mini_granularity` (**60 s**), historico treino tipico **2000** barras micro M1 |
| `deep_learning` | `arch`, `lookback` (**480**), `online_training` **false**, calibration (`neutral_half_width: 0.0`), thresholds **0.51/0.49**, `deploy_gate` |
| `orchestrator.execution` | `mandatory_trade_each_cycle: false`, `force_trade_every_cycle: false`, `invert_exec_side: false`, `scale_vision.fusion_*`, `signal_skip` 1.1, settlement **90 s** |
| `risk_management.kelly` | Stake EXPLORE (`fraction: 0.08`, piso **0.25%**, tetos stop-win Kelly ate **5%**) |
| `risk_management.soft_recovery` | RECOVER: amort **1/1**, cover **1.50**, linear3 **2.5%** |
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
  - **RECOVER** (`pending_loss > 0` ou `linear >= 1`): Soft Recovery cover pleno (`cover_multiple` **1.50**, amort **1/1**), teto `max_safe_stake_pct` — tag `RECOVER_DAL_Ln`.
- Soft Recovery cobre `pending/payout * cover_multiple` em **1** ciclo (amort **1/1**) com teto `max_safe_stake_pct` / linear3 **2.5%**.
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
- **Recovery**: Soft Recovery cover pleno (amort **1/1**, `cover_multiple` **1.50**) após loss linear; reset de risco somente quando `pending_loss` zera.
- **Loss protection**: caps edge/Z 999; quality guard em modo mandatório prioriza esteira contínua (soft alone não congela o cluster).
- **Settlement**: janela de tolerância **90 s** com reconciliação passiva (portfolio + Redis); pós-EXEC_EMPTY em recovery alinha a próxima fronteira (cap de retry).
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
