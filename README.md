# Aether Quantum Engine 2.0

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](app/.python-version)
[![Lint](https://img.shields.io/badge/Lint-ruff%20%7C%20interrogate-3776AB?logo=ruff&logoColor=white)](.github/actions/lint/action.yml)
[![Tests](https://img.shields.io/badge/Tests-pytest-0F9D58?logo=pytest&logoColor=white)](app/tests/unit)
[![Coverage](https://img.shields.io/badge/Coverage-100%25-0F9D58?logo=codecov&logoColor=white)](app/tests/unit)
[![Pre-commit](https://img.shields.io/badge/Pre--commit-active-FAB040?logo=pre-commit&logoColor=white)](.pre-commit-config.yaml)
[![CI](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/victorh-silveira/aether-quantum-engine/actions/workflows/ci.yml)

Motor quantitativo assíncrono para a Deriv: decisão por **Deep Learning (TCN PyTorch)** no par **Range Break** `RDBULL` / `RDBEAR`, velas de **5 minutos**, contratos **RISE_FALL** de **1 minuto**, dimensionamento **Kelly** com recuperação **martingale** condicionada e gates de qualidade pós-treino.

Documentação: [arquitetura](docs/arquitetura.md) | [metodologia quant](docs/medallion.md) | [estrutura do repo](docs/structure.md) | [Deriv API](docs/deriv-api.md)

Layout: `app/` (código e testes), `config/settings.json`, `docs/`, `linters/`. Ver [docs/structure.md](docs/structure.md).

---

## O que o motor faz hoje

| Etapa | Componente | Descrição |
|-------|------------|-----------|
| Dados | `StreamHandler` | WebSocket Deriv, histórico OHLC 5m, buffer configurável (padrão ~1 dia = 288 barras) |
| Decisão | `decision_bridge` + TCN | Treino walk-forward online, calibração, gating e seleção entre RDBULL/RDBEAR |
| Execução | `ExecutionManager` | Ordens RISE_FALL, cluster opcional, logs `EXEC` / `EXEC_SEL` |
| Risco | `RiskManager` | Kelly fracionário, stop win diário, martingale em recovery |
| Estado | `PersistenceManager` | `data/state.json`, checkpoints `data/dl/{symbol}.pth` |

Ciclo alinhado à vela: `orchestrator.cycle_interval_seconds: 300` (5 min).

---

## Configuração principal

Arquivo: [`config/settings.json`](config/settings.json)

| Bloco | Função |
|-------|--------|
| `symbols` / `anchor` | Universo (`RDBULL`, `RDBEAR`; âncora `RDBULL`) |
| `data_handler` | `granularity: 300`, `history_bars: 288`, `fetch_count`, `buffer_limit` |
| `deep_learning` | TCN, `lookback`, `training_history_bars`, gating, `deploy_gate` |
| `orchestrator.execution` | `mandatory_trade_each_cycle`, `invert_dl_direction` |
| `risk_management` | Kelly, martingale, stop win, stakes |
| `trading` | `demo` / `live`, janela de sessão UTC (opcional) |

Variáveis de ambiente na raiz (`.env`): token Deriv (`AETHER_DEMO_TOKEN` ou `AETHER_LIVE_TOKEN`).

---

## Gerenciamento de risco

- **Kelly fracionário** com win rate dinâmico e tetos `max_stake_pct`.
- **Stop win diário** por percentual da banca inicial (conta grande) ou valor fixo (conta pequena).
- **Martingale de recovery** quando há perda pendente no cluster, sujeito a `martingale_force_on_pending_loss` e métricas DL (`deploy_ok`, Brier, `gate_reason`).
- Cooldown por símbolo após sequência de losses (`symbol_loss_cooldown_candles`).

---

## Observabilidade

Logs em `logs/engine.log` (formato `AetherFormatter`):

- `CFG decisao` — modo DL, lookback, janela de treino, execução obrigatória, inversão
- `DL` / `DL_TRAIN` — treino, deploy, bloqueios
- `EXEC`, `EXEC_SEL`, `EXEC_NONE` — decisão e stake por ciclo
- `MARTINGALE`, `RISK` — sizing e recovery
- Liquidação e resumo de cluster após settlement

Monitor opcional: `python app/scripts/monitor/live_monitor.py`

---

## Stack e qualidade

- **Python 3.13**, `asyncio`, NumPy, Polars, PyTorch (TCN)
- **Deriv** WebSocket API v3 (`api_config.base_url` em settings)
- **CI / pre-commit**: Ruff, Interrogate, Vulture, limite 300 linhas/arquivo, pytest com **100%** de cobertura em `app/src`

Comandos (WSL recomendado):

```bash
make install
make test
make lint
make run
```

Pre-commit: `pre-commit run --all-files` (na raiz do repo).

---

## Execução ao vivo

1. Configure `.env` com o token Deriv.
2. `make install`
3. Valide checkpoints DL (seção abaixo).
4. `make run` ou `python run.py`

O motor exige `deep_learning.enabled: true`. Não há modo de decisão por LLM no pipeline ao vivo atual.

---

## Deep Learning — validação pré-live

Antes de operar com dinheiro real ou demo prolongado:

1. Remova checkpoints incompatíveis com a versão atual de features (`FEATURE_DIM` v3), se necessário.
2. Walk-forward offline por símbolo:

```bash
cd app
../app/.venv-wsl/bin/python scripts/backtest/dl_walkforward.py --symbol RDBULL
../app/.venv-wsl/bin/python scripts/backtest/dl_walkforward.py --symbol RDBEAR
```

3. Só confie no live se `deploy_ok` e métricas do relatório estiverem dentro de `deploy_gate` em `settings.json`.

Com `deploy_gate.enabled: true`, execução bloqueada com `gate_reason=deploy` quando o mini walk-forward pós-treino reprova o modelo.

**Janela de treino padrão:** 288 barras (1 dia em velas de 5m), configurável via `training_history_bars` ou `training_history_days`.

---

## Backtest legado (Medallion OTC / Gemini)

Scripts em `app/scripts/backtest/` (`medallion_backtest.py`, coleta Gemini, HFT) referem-se ao pipeline histórico **EURUSD + índices OTC + LLM**. Não são o caminho de decisão do motor ao vivo atual. Use `dl_walkforward.py` para validar o par RD.

---

## Referências

- [docs/arquitetura.md](docs/arquitetura.md) — fluxos técnicos
- [docs/medallion.md](docs/medallion.md) — filosofia quant e parâmetros de qualidade
- [docs/deriv-api.md](docs/deriv-api.md) — API Deriv (referência + uso no Aether)
- [docs/CHANGELOG.md](docs/CHANGELOG.md) — histórico de releases
