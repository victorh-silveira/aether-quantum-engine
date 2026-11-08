# Arquitetura

`PublicDeriv` descobre os instrumentos com `active_symbols` e busca candles M5/M15/H1 diretamente com `ticks_history` no WebSocket público. O domínio filtra `market=synthetic_index` pelas famílias Volatility `R_<número>` e `1HZ<número>V` e remove velas em formação. A agregação M5 serve apenas para conferir barras M15/H1 na sobreposição. `IndicatorService` gera uma leitura por vela fechada e período e grava previsões de candidatas em tabela separada. O modelo em JSON é independente por `(symbol, period, version)`; ausência ou incompatibilidade resulta em `SEM SINAL`.

`SignalStore` grava catálogo, leituras, direção observada após o próximo fechamento e o preço provisório da vela em andamento no PostgreSQL/Timescale. `Metrics` expõe estado atual ao Prometheus. Grafana usa PostgreSQL para histórico real e Prometheus para disponibilidade da coleta. O serviço indicador roda no Compose e não possui adapter de negociação nem autenticação.

Ao iniciar, o serviço restaura do banco a última leitura emitida por ativo/período. Assim, uma reinicialização no meio da vela não cria uma segunda previsão com outro checkpoint para o mesmo instante. Se o treino trocou a versão, o estado atual aguarda a próxima vela fechada antes de publicar nova leitura.

No Grafana, as visões de sinais filtram a última leitura de cada ativo/período por lado CALL/PUT, estado `ok` (APROVADO), frescor e versão do modelo ainda ativa em `indicator_model_versions`. Modelos experimentais não aparecem como sinais ativos; no catálogo, ficam como `SEM SINAL` com o estado de validação visível. O serviço atualiza esse registro a cada ciclo e o comando de treino o atualiza ao concluir. A visão de catálogo lista todos os índices ativos e mostra explicitamente os pares sem sinal. `candle_epoch` é o início da vela; o fechamento exibido soma o período, e o horizonte previsto soma dois períodos.

`config/settings.json` fixa `catalog_scope=volatility_indices`; as entradas de treino e execução rejeitam outro escopo para manter o catálogo, os modelos e os dashboards coerentes.

A coleta roda no serviço `indicator` a cada 30 segundos e atualiza o catálogo a cada seis horas. Cada previsão confirmada só muda no fechamento de uma nova vela do seu período. A tabela `indicator_live` recebe uma observação provisória da vela em andamento, comparada ao preço de referência do sinal; não é uma nova previsão. Falha de um ativo não interrompe os demais. Prometheus marca frescor separadamente; timestamps antigos não são promovidos a leituras novas.
