# Aether Quantum Engine 2.0

[![Python](https://img.shields.io/badge/Python-3.13.12-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](app/tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono para a Deriv: decisão exclusiva por **Deep Learning** (TCN, LSTM ou GRU via PyTorch) nos símbolos **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`), contratos **RISE_FALL** de **180 segundos**, classificação binária Rise/Fall com referência de confiança **0.53 / 0.47**, resolução direcional inteligente CALL/PUT, gate de **qualidade** pós-resolução e dimensionamento **Kelly** com recuperação **martingale** quando há perda pendente.

A operação divide-se em duas fases: **FASE TREINO** (nenhuma ordem até todos os modelos concluírem o treino da sessão) e **FASE OPERACAO** (operação seletiva — só entra quando o candidato passa nos pisos de qualidade; sem trade obrigatório por ciclo).

Documentação: [arquitetura](docs/arquitetura.md) | [metodologia quant](docs/medallion.md) | [estrutura do repo](docs/structure.md) | [Deriv API](docs/deriv-api.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` + `TickBuffer` | WebSocket Deriv, OHLC 180 s, ticks agregados por barra (microestrutura) |
| Fases | `_training_phase_gate` | Suspende a operação até todos os modelos concluírem o treino da sessão |
| Predição DL | `decision_bridge` + TCN/LSTM/GRU | 19 features, labels `ma_trend`, treino walk-forward deferido; predição sempre técnica OK quando dados válidos |
| Direção | `execution_direction_resolver` | Scoring composto CALL vs PUT (DL, trend, exaustão, regime); nunca abstém por conflito de indicador |
| Qualidade | `execution_quality_gate` | Filtra candidatos por score, edge, margem direcional, ADX e inversão |
| Execução | `ExecutionManager` + `execution_collect` | Ranking por `market_decision_score`; pula ciclo se nenhum candidato passa no piso |
| Risco | `RiskManager` | Kelly fracionário, stop win diário, martingale em recovery, cooldown por símbolo |
| Estado | `StateManager` + `PersistenceManager` | `data/session_state.json`, `data/state.json`, checkpoints `data/dl/{symbol}.pth` + TorchScript `_ts.pt` |

Ciclo do orquestrador: `orchestrator.cycle_interval_seconds` (padrão 60 s). Granularidade OHLC: `data_handler.granularity` (180 s). Contrato: `risk_management.params.duration` (180 s).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (`R_10` … `R_100`; âncora `R_10`) |
| `data_handler` | `granularity`, `history_bars`, `fetch_count`, `buffer_limit` |
| `deep_learning` | `arch`, `lookback`, `calibration`, `confidence_call/put_threshold`, `min_val_accuracy`, `min_edge_execute`, `deploy_gate` |
| `orchestrator.execution` | `direction_scoring`, `dynamic_threshold`, `quality_gate`, `mandatory_trade_each_cycle`, settlement |
| `risk_management.kelly` | Kelly, martingale, `mandatory_min_trade_score`, recovery floors |
| `risk_management.params` | `duration: 180`, stakes |
| `trading` | `demo` / `live` |

Variáveis na raiz (`.env`): `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` (opcional). Validação: `python app/scripts/operations/deriv_pat_connect.py`.

---

## Gerenciamento de risco

- **Kelly fracionário** com win rate dinâmico e tetos `max_stake_pct`.
- **Stop win diário** por percentual da banca inicial (conta grande) ou valor fixo (conta pequena).
- **Martingale de recovery** quando há perda pendente: stake cobre perda integral + alvo derivado do payout, limitada por banca e `stake_max`.
- Cooldown por símbolo após sequência de losses (`symbol_loss_cooldown_candles`).

---

## Fases, recovery e execução

- **FASE TREINO**: ao iniciar a sessão, todo símbolo retreina pelo menos uma vez. Enquanto qualquer modelo não concluir, nenhuma ordem é enviada.
- **FASE OPERACAO** com `mandatory_trade_each_cycle: false`: o motor opera apenas quando o melhor candidato passa no **gate de qualidade** (score ≥ 0.68 normal, pisos recovery mais altos com perdas consecutivas).
- **Bloqueio absoluto** somente para falhas técnicas: `data`, `predict_error`, `training`, `deploy_ok=false`.
- **Recovery**: ranking de mercado com diversificação de símbolo; martingale com convicção mínima 0.64 e `val_accuracy` ≥ 0.62.

---

## Observabilidade

Logs em `logs/engine.log` (formato `AetherFormatter`):

- `CFG decisao` — modo DL, lookback, histórico de treino, execução obrigatória
- `FASE TREINO` / `FASE OPERACAO` — transição entre fases
- `DL` / `DL REC` — linha curta: `exec`, `bias` (ajuste direcional), `skip` (bloqueio técnico)
- `EXEC_SEL` — símbolo escolhido com `ord=`, `dl=`, métricas `s`/`v`/`r`, indicadores e alternativas
- `SKIP` — ciclo pulado por piso de qualidade
- `MARTINGALE`, `RISK: RECOVERY` — sizing e recovery
- Liquidação e resumo de cluster após settlement

Mensagens repetidas são deduplicadas (`log_dedupe`). Cada ciclo e bloco de treino são separados por linha em branco.

Monitor opcional: `python app/scripts/monitor/live_monitor.py`

---

## Stack e qualidade

- **Python 3.13.12**, `asyncio`, NumPy, Polars, PyTorch (TCN / LSTM / GRU)
- **Deriv** PAT + REST OTP + WebSocket (`api_config` em settings; ver `docs/deriv-api.md`)
- **CI / pre-commit**: Ruff, Interrogate, Vulture, limite 300 linhas/arquivo, pytest com **100%** de cobertura em `app/src`

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
3. Treine os modelos: `python train.py` ou `app/scripts/batch/launch-train.bat`.
4. Valide checkpoints DL em `data/dl/`.
5. Execute o motor: `python run.py`, `make run` ou `launch-all-demo.bat` / `launch-all-live.bat`.

O motor exige `deep_learning.enabled: true` e checkpoints válidos em `data/dl/`. Treino e execução são processos separados — `train.py` grava os modelos; `run.py` só opera.

---

## Referências

- [docs/arquitetura.md](docs/arquitetura.md) — fluxos técnicos
- [docs/medallion.md](docs/medallion.md) — filosofia quant e parâmetros de qualidade
- [docs/deriv-api.md](docs/deriv-api.md) — API Deriv
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — histórico de releases
