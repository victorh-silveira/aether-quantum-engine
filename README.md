# Aether Quantum Engine 2.0

[![Python](https://img.shields.io/badge/Python-3.13.12-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](app/tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono para a Deriv: decisão por **Deep Learning (TCN PyTorch)** nos símbolos **Range Break** (`R_10`, `R_25`, `R_50`, `R_75`, `R_100`), contratos **RISE_FALL** de **1 minuto**, dimensionamento **Kelly** e recuperação **martingale** integral quando há perda pendente.

A operação é dividida em duas fases: **FASE TREINO** (nenhuma ordem é enviada até que todos os modelos concluam o primeiro treino válido) e **FASE OPERACAO** (um trade por ciclo via ranking de mercado, com recovery inteligente).

Documentação: [arquitetura](docs/arquitetura.md) | [metodologia quant](docs/medallion.md) | [estrutura do repo](docs/structure.md) | [Deriv API](docs/deriv-api.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` | WebSocket Deriv, histórico OHLC, buffer configurável |
| Fases | `_training_phase_gate` | Suspende a operação até todos os modelos concluírem o primeiro treino (`FASE TREINO` → `FASE OPERACAO`) |
| Decisão | `decision_bridge` + TCN | Treino walk-forward em background (slot único com prioridade para modelos não treinados), calibração, gating |
| Execução | `ExecutionManager` | Modo obrigatório: uma ordem por ciclo escolhida por ranking de mercado (`execution_market_rank`); logs `EXEC` / `EXEC_SEL` |
| Risco | `RiskManager` | Kelly fracionário, stop win diário, martingale em recovery, cooldown por símbolo |
| Estado | `PersistenceManager` | `data/state.json`, checkpoints `data/dl/{symbol}.pth` |

Ciclo do orquestrador: `orchestrator.cycle_interval_seconds` (padrão 60 s). Granularidade OHLC: `data_handler.granularity` (padrão 60 s).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (`R_10` … `R_100`; âncora `R_50`) |
| `data_handler` | `granularity`, `history_bars`, `fetch_count`, `buffer_limit` |
| `deep_learning` | TCN, `lookback`, `training_history_bars`, gating, `deploy_gate`, `recovery_gating` |
| `orchestrator.execution` | `mandatory_trade_each_cycle`, `diversify_after_loss_margin`, settlement |
| `risk_management` | Kelly, martingale, stop win, stakes |
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

- **FASE TREINO**: enquanto qualquer modelo não tiver o primeiro treino válido (`val_brier` acima de `brier_untrained_floor`), nenhuma ordem é enviada e o slot de treino em background fica exclusivo para os modelos pendentes. A transição é registrada uma única vez (`FASE OPERACAO || todos os modelos treinados`).
- **FASE OPERACAO** com `mandatory_trade_each_cycle: true`: o motor envia uma ordem por ciclo, escolhida por ranking de mercado (score calibrado, convicção bruta, val_accuracy, edge, Brier, deploy e alinhamento com o contexto binário da última vela).
- Direção de execução: sinal DL refinado por `resolve_market_direction` — quando a convicção bruta é fraca, extremos estatísticos da vela (`sma_z`) aplicam reversão à média.
- **Recovery**: prioriza direção alinhada ao último loss com pisos de qualidade (`recovery_gating`), diversifica o símbolo (bônus por não repetir o par perdedor) e usa o núcleo `R_75`/`R_50`.
- Bloqueios duros nunca são forçados, nem no modo obrigatório: `training`, `cooldown`, `session_pause`, `data` e `predict_error`.

---

## Observabilidade

Logs em `logs/engine.log` (formato `AetherFormatter`):

- `CFG decisao` — modo DL, lookback, histórico de treino, execução obrigatória
- `FASE TREINO` / `FASE OPERACAO` — transição entre fase de treinamento e fase de operação
- `DL` / `DL_TREINO` — resumo de decisões por ciclo, modelos em primeiro treino
- `EXEC`, `EXEC_SEL`, `EXEC_NONE` — decisão, alternativas e stake por ciclo (com métricas `s`/`v`/`r`/`b`)
- `MARTINGALE`, `RISK: RECOVERY` — sizing e recovery
- Liquidação e resumo de cluster após settlement

Mensagens repetidas são deduplicadas (`log_dedupe`): só voltam ao nível `INFO` quando o conteúdo muda. Ciclos com trade são separados por linha em branco.

Monitor opcional: `python app/scripts/monitor/live_monitor.py`

---

## Stack e qualidade

- **Python 3.13.12**, `asyncio`, NumPy, Polars, PyTorch (TCN)
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

O motor exige `deep_learning.enabled: true`. Não há modo de decisão por LLM no pipeline ao vivo.

---

## Referências

- [docs/arquitetura.md](docs/arquitetura.md) — fluxos técnicos
- [docs/medallion.md](docs/medallion.md) — filosofia quant e parâmetros de qualidade
- [docs/deriv-api.md](docs/deriv-api.md) — API Deriv
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — histórico de releases
