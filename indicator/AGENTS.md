# AGENTS.md — Aether Synthetic Indicator

## Estrategia operacional vigente

O servico publico emite o setup fixo `pullback-1`: EMA20/EMA50, inclinacao de tres velas, recuo ate 0,5 ATR14 e confirmacao pela maxima/minima do recuo com corpo de 0,2 a 1,5 ATR14. Usa somente velas fechadas e continuas de M15/H1/D1; somente M15 pode gerar ordem no motor. `SEM SINAL` traz motivo verificavel. `make app-train` coleta dados e informa apenas proxy direcional de velas; nao treina ou promove modelos. O historico de ML abaixo e pesquisa legada, sem autoridade para contratos. A versao do setup fica em `pullback_registry`; 200 liquidacoes DEMO de `setup_pilot` e limite inferior unilateral de 95% positivo promovem a versao. REAL usa posterior Beta(0,5;0,5), haircut e EV positivo na proposta final. O cliente publico nao compra contratos.

- Responder e documentar em PT-BR. Trabalhar no WSL com Python 3.13.
- Produto: observação dos índices Volatility e Ações OTC da Deriv em M15, H1 e D1. Forex é operação manual externa; não há inferência cruzada.
- Serviço ativo no Compose: `app/src/indicator/`, entrada `app/run.py`, treino `app/train.py`, configuração `config/settings.json`.
- Selecionar somente `R_<número>` e `1HZ<número>V` ativos no catálogo sintético. Somente `active_symbols` e `ticks_history` no cliente público. Nunca incluir `buy`, `proposal`, `authorize`, liquidação, Kelly, recuperação ou credenciais de conta.
- `SEM SINAL` é obrigatório sem histórico contínuo, dado atual ou modelo próprio. Não reutilizar o último lado. Distinguir sempre `APROVADO` (supera maioria em acurácia e Brier) de `EXPERIMENTAL` (modelo avaliado sem ganho nos dois critérios); este último nunca deve ser exibido como sinal CALL/PUT ativo nos dashboards.
- Modelo por ativo e período, validação temporal contra maioria em acurácia e Brier. Histórico observado é direção da vela seguinte, não resultado de contrato.
- O treino compara regularizacoes em bloco cronologico, calibra no bloco seguinte e valida no bloco final. `OK` e uma aprovacao estatistica; `ev_candidatas=0` indica que nenhuma probabilidade superou o break-even de referencia. Somente liquidacoes confirmadas do broker no motor podem promover uma versao para REAL.
- A previsão confirmada usa apenas velas fechadas e não é reescrita dentro do período. Preço e direção da vela em andamento podem ser exibidos como observação provisória, com timestamp e sem alegar acerto antecipado.
- Relatar métricas da validação por par; arquivar checkpoint anterior se uma nova avaliação reprovar esse par.
- Domínio puro; I/O em adapters; nenhuma interpolação de dados. Arquivos Python em `app/src` com no máximo 300 linhas; 100% de cobertura de linhas.
- Indicador e motor usam um unico Compose, TimescaleDB, Prometheus e Grafana em `localhost:3000` com credencial `admin`/`aether`. Execute `make docker-up` ou `make app-train` na raiz.
- Os dashboards de sinais mostram apenas pares com última leitura CALL/PUT válida e atual; o dashboard de catálogo lista todos os índices ativos, inclusive os sem sinal.
- Os dashboards de sinais exigem versão do modelo ainda ativa em `indicator_model_versions`; `candle_epoch` representa abertura da vela, e o fechamento previsto ocorre em `candle_epoch + 2 * período`.
- Atualizar código, testes, documentação, dashboards, regras e skills em conjunto. Executar `make app-lint` e `make app-test` na raiz para validar a matriz completa. Preservar a interface `app-*`/`docker-*` da `main` no Makefile, com `make help` como alvo padrão.
- Na `main`, este diretório permanece público e observacional. O motor autenticado em `../app/` consulta apenas sinais M15 aprovados; o lado aprovado determina a ordem mesmo quando a TCN diverge. Não introduzir compras neste subsistema. O CI principal valida este código antes de publicar o motor.

Ver [arquitetura](docs/arquitetura.md), [operação](docs/operacao.md), [modelos](docs/modelos.md) e [matriz de agentes](docs/agent-coverage.md).
