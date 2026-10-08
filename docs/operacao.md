# Operação e observabilidade

1. `make infra-up` sobe indicador, Timescale, Prometheus e Grafana em portas locais separadas do motor antigo.
2. `.venv/bin/python app/train.py` tenta treinar todos os ativos sintéticos do catálogo público. Os modelos são montados no serviço e carregados no ciclo seguinte.
3. No Grafana (`http://localhost:13000`), use **Visao geral** para localizar `SEM SINAL`/dados atrasados e **Ativo** para escolher símbolo e período.

O serviço `indicator` expõe `http://localhost:9101/metrics`; o Prometheus em `http://localhost:19090` coleta `indicator:9101` pela rede interna. Estados: `ok`, `modelo_indisponivel`, `historico_insuficiente`, `dados_atrasados`. Falha de catálogo vazio interrompe o ciclo; falha isolada de histórico é registrada por ativo. Se o alvo Prometheus estiver DOWN, confira `docker compose -f infra/docker/docker-compose.yml ps` e os logs do serviço `indicator`.

As tabelas `indicator_symbols` e `indicator_signals` são criadas na primeira inicialização. A coluna `observed_side` é preenchida após o próximo fechamento. Lacunas permanecem lacunas. O painel histórico usa timestamps gravados e pontos isolados, sem interpolação.
