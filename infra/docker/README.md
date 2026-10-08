# Infraestrutura do indicador

`docker compose -f infra/docker/docker-compose.yml up -d` sobe apenas Timescale, Prometheus e Grafana. Não há Redis, MinIO ou serviços de execução de contratos. O Python do indicador roda no host. O Prometheus lê `host.docker.internal:9101`; o Compose mapeia o gateway do host para Linux. As portas de banco e painéis ficam em loopback.
