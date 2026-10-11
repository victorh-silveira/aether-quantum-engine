# Infraestrutura do indicador

`make docker-up` sobe indicador, Timescale, Prometheus e Grafana. `make docker-smoke` valida os endpoints; `make docker-logs DOCKER_SERVICE=grafana` mostra os logs desse serviço. Não há Redis, MinIO ou serviços de execução de contratos. O Prometheus lê `indicator:9101` pela rede interna. O diretório local `data/indicator-models` é montado somente para leitura no serviço. As portas de banco e painéis ficam em loopback.
