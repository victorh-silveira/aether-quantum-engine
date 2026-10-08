# Aether Synthetic Indicator

Indicador observacional dos **mercados sintéticos** retornados pelo catálogo público da Deriv. Exibe CALL, PUT ou **SEM SINAL** por ativo em M5, M15 e H1. Não envia propostas, não compra contratos e não usa autenticação de conta. Sinais de sintéticos não são previsões para pares Forex.

## Uso no WSL

Requer Python 3.13, Docker Compose e acesso ao WebSocket público da Deriv.

```bash
python -m venv .venv
.venv/bin/pip install -r app/requirements-dev.txt
make infra-up
.venv/bin/python app/train.py
```

O treino percorre o catálogo atual e grava um modelo por ativo/período em `data/indicator-models/` somente quando sua validação temporal supera a referência de maioria em acurácia e Brier. Pares sem modelo aprovado aparecem como **SEM SINAL**. O processo de treino pode demorar conforme o tamanho do catálogo e os limites da API. Reexecute após entrada de novos ativos ou para atualizar modelos.

Grafana: `http://localhost:13000`, dashboards **Sinteticos | Visao geral** e **Sinteticos | Ativo**. Prometheus: `http://localhost:19090`. Métricas do serviço indicador: `http://localhost:9101/metrics`.

O motor usa apenas velas M5 fechadas; M15 e H1 são agregadas somente quando todas as velas M5 do bloco estão presentes. Cada leitura inclui horário da vela, probabilidade CALL, versão do modelo e motivo. O resultado observado é preenchido após o fechamento da vela seguinte; representa direção de mercado, não resultado de contrato. Sem dados atuais ou histórico suficiente, aparece **SEM SINAL**.

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

`make infra-up` inicia o indicador junto com a stack. O treino é offline; os modelos exportados ficam visíveis ao serviço pelo volume local e são carregados no ciclo seguinte. `python run.py` serve apenas ao desenvolvimento quando o serviço Compose estiver parado.
