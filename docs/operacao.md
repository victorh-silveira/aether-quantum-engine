# Operação e observabilidade

1. `make infra-up` sobe indicador, Timescale, Prometheus e Grafana em portas locais separadas do motor antigo.
2. `.venv/bin/python app/train.py` tenta treinar todos os ativos sintéticos do catálogo público. Os modelos são montados no serviço e carregados no ciclo seguinte.
3. No Grafana (`http://localhost:13000`), use **Visao geral** para localizar `SEM SINAL`/dados atrasados e **Ativo** para escolher símbolo e período.

Os dois dashboards referenciam o datasource Timescale pelo UID `timescale` e pelo tipo interno `grafana-postgresql-datasource`. O arquivo de provisionamento usa `type: postgres`, que o Grafana converte para esse identificador interno. Se aparecer "Datasource timescale was not found", confira o UID e o tipo retornados por `/api/datasources`, além dos JSONs dos painéis.

As consultas históricas filtram `candle_epoch` (segundos Unix) com `$__unixEpochFilter(candle_epoch)`. O macro `$__timeFilter` aceita uma coluna de data/hora; não deve envolver `to_timestamp(candle_epoch)`, pois o Grafana expande a expressão incorretamente.

O serviço `indicator` expõe `http://localhost:9101/metrics`; o Prometheus em `http://localhost:19090` coleta `indicator:9101` pela rede interna. Estados: `ok`, `modelo_indisponivel`, `historico_insuficiente`, `dados_atrasados`. Falha de catálogo vazio interrompe o ciclo; falha isolada de histórico é registrada por ativo. Se o alvo Prometheus estiver DOWN, confira `docker compose -f infra/docker/docker-compose.yml ps` e os logs do serviço `indicator`.

As tabelas `indicator_symbols` e `indicator_signals` são criadas na primeira inicialização. A coluna `observed_side` é preenchida após o próximo fechamento. Lacunas permanecem lacunas. O painel histórico usa timestamps gravados e pontos isolados, sem interpolação.
