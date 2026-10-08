# Infraestrutura do indicador

`docker compose -f infra/docker/docker-compose.yml up -d` sobe indicador, Timescale, Prometheus e Grafana. Não há Redis, MinIO ou serviços de execução de contratos. O Prometheus lê `indicator:9101` pela rede interna. O diretório local `data/indicator-models` é montado somente para leitura no serviço. As portas de banco e painéis ficam em loopback.
