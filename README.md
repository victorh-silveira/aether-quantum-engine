# Aether Quantum Engine 2.0

[![Python](https://img.shields.io/badge/Python-3.13.12-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](app/tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono para a Deriv: decisão exclusiva por **Deep Learning** (TCN, LSTM ou GRU via PyTorch) nos símbolos **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`), contratos **RISE_FALL** de **60 segundos**, classificação binária Rise/Fall com threshold **0.75 / 0.25**, dimensionamento **Kelly** e recuperação **martingale** integral quando há perda pendente.

A operação é dividida em duas fases: **FASE TREINO** (nenhuma ordem é enviada até que todos os modelos concluam o treino da sessão) e **FASE OPERACAO** (operação seletiva — só entra quando o modelo atinge alta confiança; sem trade obrigatório por ciclo).

Documentação: [arquitetura](docs/arquitetura.md) | [metodologia quant](docs/medallion.md) | [estrutura do repo](docs/structure.md) | [Deriv API](docs/deriv-api.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` + `TickBuffer` | WebSocket Deriv, OHLC 60 s, ticks agregados por barra (microestrutura) |
| Fases | `_training_phase_gate` | Suspende a operação até todos os modelos concluírem o treino da sessão |
| Decisão | `decision_bridge` + TCN/LSTM/GRU | 19 features, labels binários (horizon 1 barra), treino walk-forward deferido, gating 0.75/0.25 |
| Execução | `ExecutionManager` | Operação seletiva: só entra com `raw_prob >= 0.75` ou `<= 0.25`; ranking entre candidatos elegíveis |
| Risco | `RiskManager` | Kelly fracionário, stop win diário, martingale em recovery, cooldown por símbolo |
| Estado | `PersistenceManager` | `data/state.json`, checkpoints `data/dl/{symbol}.pth` + TorchScript `_ts.pt` |

Ciclo do orquestrador: `orchestrator.cycle_interval_seconds` (padrão 3 s). Granularidade OHLC: `data_handler.granularity` (60 s). Contrato: `risk_management.params.duration` (60 s).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (`R_10` … `R_100`; âncora `R_50`) |
| `data_handler` | `granularity`, `history_bars`, `fetch_count`, `buffer_limit` |
| `deep_learning` | `arch` (tcn/lstm/gru), `lookback`, `confidence_call_threshold`, `confidence_put_threshold`, `min_val_accuracy`, `deploy_gate` |
| `orchestrator.execution` | `mandatory_trade_each_cycle` (false), `diversify_after_loss_margin`, settlement |
| `risk_management.kelly` | Kelly, martingale |
| `risk_management.params` | `duration: 60`, stakes |
| `trading` | `demo` / `live` |

Variáveis na raiz (`.env`): `AETHER_DERIV_PAT`, `AETHER_DERIV_APP_ID`, `AETHER_DERIV_ACCOUNT_ID` (opcional). Validação: `python app/scripts/deriv_pat_connect.py`.

---

## Gerenciamento de risco

- **Kelly fracionário** com win rate dinâmico e tetos `max_stake_pct`.
- **Stop win diário** por percentual da banca inicial (conta grande) ou valor fixo (conta pequena).
- **Martingale de recovery** quando há perda pendente: stake cobre perda integral + alvo derivado do payout, limitada por banca e `stake_max`.
- Cooldown por símbolo após sequência de losses (`symbol_loss_cooldown_candles`).

---

## Fases, recovery e execução

- **FASE TREINO**: ao iniciar a sessão, todo símbolo retreina pelo menos uma vez. Enquanto qualquer modelo não concluir esse treino, nenhuma ordem é enviada.
- **FASE OPERACAO** com `mandatory_trade_each_cycle: false`: o motor opera apenas quando `raw_prob >= 0.75` (CALL) ou `<= 0.25` (PUT). Ciclos sem sinal forte são pulados.
- **Threshold de confiança**: piso de deploy `min_val_accuracy: 0.53` (lucratividade mínima vs payout ~95%).
- **Recovery**: ranking de mercado com diversificação de símbolo e hedge no par Range quando aplicável.
- Bloqueios duros: `training`, `cooldown`, `session_pause`, `data`, `predict_error`, `deploy`.

---

## Observabilidade

Logs em `logs/engine.log` (formato `AetherFormatter`):

- `CFG decisao` — modo DL, lookback, histórico de treino, execução obrigatória
- `FASE TREINO` / `FASE OPERACAO` — transição entre fase de treinamento e fase de operação
- `DL` / `DL TREINO` — resumo por ciclo; progresso por época (`iniciado`, `epoca X/Y`, `concluido`) com blocos separados por linha em branco
- `EXEC`, `EXEC_SEL`, `EXEC_NONE` — decisão, alternativas e stake por ciclo (com métricas `s`/`v`/`r`/`b`)
- `MARTINGALE`, `RISK: RECOVERY` — sizing e recovery
- Liquidação e resumo de cluster após settlement

Mensagens repetidas são deduplicadas (`log_dedupe`): só voltam ao nível `INFO` quando o conteúdo muda. Cada ciclo, cada treino de par e cada bloco de treino vs operação são separados por linha em branco (`BlankLineSquasher` evita linhas vazias consecutivas).

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
python app/scripts/deriv_pat_connect.py
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
3. Valide checkpoints DL em `app/data/dl/`.
4. `make run` ou `launch-all-demo.bat` / `launch-all-live.bat`

O motor exige `deep_learning.enabled: true`. Checkpoints antigos (FEATURE_DIM diferente de 19 ou versão < 4) são invalidados — o bootstrap retreina na primeira execução. Não há modo de decisão alternativo no pipeline ao vivo.

---

## Referências

- [docs/arquitetura.md](docs/arquitetura.md) — fluxos técnicos
- [docs/medallion.md](docs/medallion.md) — filosofia quant e parâmetros de qualidade
- [docs/deriv-api.md](docs/deriv-api.md) — API Deriv
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — histórico de releases
