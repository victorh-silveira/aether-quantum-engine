# Arquitetura

`PublicDeriv` descobre os instrumentos com `active_symbols` e busca candles M5 com `ticks_history` no WebSocket público. O domínio filtra apenas `market=synthetic_index`, remove velas em formação e agrega blocos M15/H1 completos. `IndicatorService` gera uma leitura por vela fechada e período. O modelo em JSON é independente por `(symbol, period)`; ausência ou incompatibilidade resulta em `SEM SINAL`.

`SignalStore` grava catálogo, leituras e direção observada após o próximo fechamento no PostgreSQL/Timescale. `Metrics` expõe estado atual ao Prometheus. Grafana usa PostgreSQL para histórico real e Prometheus para disponibilidade da coleta. O serviço indicador roda no Compose e não possui adapter de negociação nem autenticação.

A coleta roda no serviço `indicator` a cada abertura M5 e atualiza o catálogo a cada seis horas. Falha de um ativo não interrompe os demais. Prometheus marca frescor separadamente; timestamps antigos não são promovidos a leituras novas.
