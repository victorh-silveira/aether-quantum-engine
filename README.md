# Aether Synthetic Indicator

Indicador observacional dos **índices Volatility** retornados pelo catálogo público da Deriv. Exibe CALL, PUT ou **SEM SINAL** por ativo em M5, M15 e H1. Não envia propostas, não compra contratos e não usa autenticação de conta. Sinais de sintéticos não são previsões para pares Forex.

## Uso no WSL

Requer Python 3.13, Docker Compose e acesso ao WebSocket público da Deriv. `make` abre o menu de ajuda, como na `main`; os targets `app-*` e `docker-*` usam a mesma interface com ações próprias do indicador.

```bash
python -m venv .venv
.venv/bin/pip install -r app/requirements-dev.txt
make docker-up
make app-train
```

O treino percorre somente as famílias `R_<número>` e `1HZ<número>V` ativas no catálogo público. Cada par com pelo menos 200 amostras recebe um modelo próprio em `data/indicator-models/`. A validação temporal distingue **APROVADO** (superou a maioria em acurácia e Brier) de **EXPERIMENTAL** (não superou ambos). O segundo estado permanece no catálogo como **SEM SINAL**; não é exibido como CALL/PUT ativo. Sem modelo, dados atuais ou histórico suficiente, permanece **SEM SINAL**. O terminal mostra progresso por ativo, página e período; requisições e ativos têm prazos limitados.

Grafana: `http://localhost:3000` (usuário `admin`, senha `aether`). **Sinteticos | Leituras atuais** e **Sinteticos | Ativo** exibem somente sinais CALL/PUT atuais de modelos **APROVADOS**, com movimento provisório; **Sinteticos | Todos os indices** mostra o catálogo completo, inclusive `SEM SINAL`. Prometheus: `http://localhost:19090`. Métricas do serviço indicador: `http://localhost:9101/metrics`.

O motor recalcula a previsão com velas fechadas: M5 a cada 5 minutos, M15 a cada 15 minutos e H1 a cada hora. M15 e H1 são agregadas somente quando todas as velas M5 do bloco estão presentes. A cada 30 segundos, uma observação provisória do preço em formação mostra se o movimento atual está **A FAVOR** ou **CONTRA** o sinal emitido; ela não altera a previsão histórica. Cada leitura inclui horário da vela, horizonte de fechamento, probabilidade CALL, versão do modelo e validação. O resultado observado é preenchido após o fechamento da vela seguinte; representa direção de mercado, não resultado de contrato. Sem dados atuais ou histórico suficiente, aparece **SEM SINAL**.

```bash
make lint
make test
```

Esta branch mantém a estrutura do CI/CD anterior (Python, Docker, Shell, Rust,
workflows e resumo), com comandos adaptados ao indicador. Valida push e PRs
destinados à `codex/synthetic-indicator`. Push aprovado publica imagem de prévia
no GHCR, sem release semântico ou deploy. A `main` mantém o motor e CI/CD
anteriores.
Os detalhes do pipeline estão em [WORKFLOWS.md](.github/workflows/WORKFLOWS.md).

`make infra-up` inicia o indicador junto com a stack. O treino é offline; os modelos exportados ficam visíveis ao serviço pelo volume local e são carregados no ciclo seguinte. `python run.py` serve apenas ao desenvolvimento quando o serviço Compose estiver parado, pois ambos usam a porta 9101.
