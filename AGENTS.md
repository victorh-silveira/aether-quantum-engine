# AGENTS.md — Aether Quantum Engine (Branch Laya Continuo R_10)

Ponto de entrada unico e soberano para agentes autonomos de IA e desenvolvedores nesta branch de refatoracao 100% orientada ao **Laya System 1**.

---

## 1. Identidade e Papel do Agente

- **O que voce e:** Copiloto senior de engenharia de software, arquitetura de sistemas e integracao de inferencia System 1 com Laya.
- **Autoridade de Decisao:** **100% da decisao binaria (`CALL` ou `PUT`)** pertence ao motor **Laya** (`convaiinnovations/laya`, `vegaml=False`) alimentado por velas M5 fechadas, pacote completo de indicadores tecnicos causais e configuracoes sanitizadas de `config/settings.json`.
- **Comportamento basico:** Zero alucinacao, rigor analitico, estrita adesao aos contratos de software e verificacao constante atraves de suites de testes automatizadas com **100% de cobertura**.

---

## 2. Ambiente Operacional e Diretrizes de Engenharia

- **Idioma oficial:** Todas as respostas, documentacoes tecnicas e mensagens de commit devem ser escritas em **PT-BR**.
- **Terminal e Scripts:** Todas as operacoes devem ser executadas no ambiente **WSL Linux (Bash)**.
- **Runtime:** Motor executado no host com **CPython 3.13.18** e **asyncio** (ambiente Conda `deriv-api`) usando **`laya==0.5.5`** e **PyTorch**.
- **Arquitetura de Software:** **DDD (Domain-Driven Design) e Hexagonal (Ports & Adapters)**. O dominio (`src/domain/`) permanece puro, sem imports de infraestrutura ou frameworks (`laya`, `torch`, `redis`, `asyncpg`).
- **DataFrames e Series:** **Polars** e o padrao para manipulacao colunar; **pandas e estritamente proibido**.
- **Integridade Visual:** Sem emojis em codigo-fonte, logs de sistema ou documentacoes tecnicas.
- **Documentacao de Codigo:** Codigo autoexplicativo com docstrings completas (100% Interrogate).
- **Contrato de Commits:** Conventional Commits em PT-BR via `scripts/wsl/commit.sh` (ou `make app-commit`).

---

## 3. Operacao Continua Laya no `R_10` M5 (Zero SKIP)

- **Ativo Unico:** **`R_10`** (Volatility 10 Index — Deriv).
- **Contrato Operacional:** Rise/Fall (`CALL`/`PUT`) com duracao fixa de **5 minutos** (`duration=5`, `duration_unit="m"`, velas M5 `period_seconds=300`).
- **Regra Absoluta de Continuidade:** Em hipotese alguma o motor deve dar `SKIP` ou ignorar um trade por veto de regime, horario, EV, streak ou stop-win. Em caso de empate exato de probabilidades (`0.5 / 0.5`), o desempate e deterministico por `ema_20 >= ema_50 -> CALL`, senao `PUT`.
- **Pacote de Indicadores Passado ao Laya (`compute_indicators`):** `EMA(9,20,21,50)`, `ema_spread_atr`, `ema20_slope`, `ema50_slope`, `RSI(7,14)`, `MACD(12,26,9)`, `Stochastic(14,3)`, `CCI(14)`, `CMO(14)`, `Williams%R(14)`, `ROC(14)`, `ADX/DI(14,20)`, `ATR(14,50)`, `atr_ratio`, `Bollinger %B/Width`, `Keltner %B`, `Hurst(14)`, `body_atr`, `upper/lower_wick_atr`, `candle_dir`, `streak` e `session_zscore`.
- **Logs Obrigatorios no `make app-run`:** Todo ciclo M5 imprime `INDICATORS | ...` (todos os indicadores calculados) e `LAYA_DECISION | ...` (`dir`, `P(CALL)`, `P(PUT)`, `prob`, `conf`, `lat`, `stake`, `banca`).
- **Integridade de Inferencia:** Usar exclusivamente o buffer M5 fechado, com pelo menos 51 barras e identidade preservada durante a inferencia. Dimensionar o contexto pelo tokenizer real e rejeitar truncamento; `LAYA_CONTEXT` registra epoch, assinatura e consumo de tokens. Nao alterar probabilidades artificialmente nem interpretar confianca como taxa de acerto financeiro validada.
- **Formatacao Numerica:** Exibir valores de indicadores, probabilidades, confianca, latencia, duracoes e dinheiro com duas casas decimais nos logs Python. Preservar precisao dos calculos, dados de inferencia, metricas Prometheus, identificadores, epochs e contagens tecnicas.

### Gestao de Capital e Dimensionamento (`kelly_recovery`):
- **Kelly Fracionado:** `f=(b*p-(1-p))/b`, multiplicador 0,25 e payout liquido cotado. Nao elevar probabilidades para fabricar edge.
- **Piso Continuo:** Manter 1% da banca e minimo $1,00 mesmo com Kelly nulo ou negativo, por mandato do operador. Registrar `FLOOR` explicitamente.
- **Recuperacao por Payout:** Adaptacao da main `460c4f41`, com divida reduzida apenas pelo lucro realizado, amortizacao em etapas e teto absoluto de 5% por ordem. Nao aplicar Martingale 2x ou progressao geometrica.
- **Treinamento GPU:** Preparar 90 dias R_10 M5, dividir temporalmente 60/10/10/20 e calibrar fora do treino. Promover apenas candidato aprovado no teste final; treinamento exige CUDA. Referencia: `docs/laya-training.md`.
- **Make e CUDA:** `make app-cuda-install` instala PyTorch CUDA; `app-cuda-check` e `app-cuda-train-check` validam inferencia e backward sem ordens. Usar `app-prepare`, `app-train`, `app-evaluate`, `app-promote` e `app-run`; perfil auto=head em GPUs de ate 12 GiB. Full exige 12 GiB livres, head exige 3 GiB livres. Runtime e treino verificam CUDA antes de iniciar pelo Makefile.

---

## 4. Proibicoes Globais e Hard Gates

1. Introduzir `SKIP`, filtros ou vetos que bloqueiem a decisao continua `CALL`/`PUT` do Laya.
2. Ultrapassar o teto de **5% da banca** por ordem, fabricar edge para Kelly ou introduzir Martingale/D'Alembert geometrico.
3. Criar ou manter arquivos em `src/**/*.py` acima de **300 linhas**.
4. Reduzir a cobertura de testes unitarios em `src` abaixo de **100%**.
5. Utilizar pandas ou inserir emojis em codigo, logs, docs ou commits.

---

## 5. Harness de Agentes de IA (`.agents/`) e Documentacao (`docs/`)

- **Regras:** [`.agents/rules/`](.agents/rules/) (`aether-laya-decision.mdc`, `aether-engineering.mdc`, `aether-domain-pure.mdc`, `aether-testing.mdc`, etc.).
- **Habilidades:** [`.agents/skills/`](.agents/skills/) (`aether-laya-engine`, `aether-architecture-senior`, `aether-precommit`, etc.).
- **Documentacao Principal:** [docs/strategy-laya.md](docs/strategy-laya.md), [docs/arquitetura.md](docs/arquitetura.md), [docs/engineering-settings-ssot.md](docs/engineering-settings-ssot.md), [docs/engineering-orchestrator.md](docs/engineering-orchestrator.md).
