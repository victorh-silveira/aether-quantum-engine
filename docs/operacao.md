# Operação e observabilidade

1. `make infra-up` sobe Timescale, Prometheus e Grafana em loopback.
2. `.venv/bin/python app/train.py` tenta treinar todos os ativos sintéticos do catálogo público.
3. `.venv/bin/python run.py` inicia coleta contínua e expõe `/metrics` na porta 9100.
4. No Grafana, use **Visao geral** para localizar `SEM SINAL`/dados atrasados e **Ativo** para escolher um símbolo e período.

Estados: `ok`, `modelo_indisponivel`, `historico_insuficiente`, `dados_atrasados`. Falha de catálogo vazio interrompe o ciclo; falha isolada de histórico é registrada por ativo. Se Prometheus estiver DOWN, confirme o processo host em `http://localhost:9101/metrics` e o mapeamento `host.docker.internal` no Compose.

As tabelas `indicator_symbols` e `indicator_signals` são criadas na primeira inicialização. A coluna `observed_side` é preenchida após o próximo fechamento. Lacunas permanecem lacunas. O painel histórico usa os timestamps gravados, sem interpolação.
