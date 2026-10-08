# Aether Quantum Engine

Motor quantitativo assíncrono para contratos Rise/Fall de 5 minutos no índice sintético `1HZ75V` da Deriv. O processo Python 3.13 roda no host com `asyncio`; Redis, TimescaleDB, MinIO, meta LightGBM e loss-classifier rodam como serviços Docker. A arquitetura separa domínio, aplicação, portas, adaptadores e apresentação. O estado tabular usa Polars.

A configuração operacional está em [`config/settings.json`](config/settings.json). O universo e a cadência atuais são M5 de 300 segundos, contexto D1 de 86.400 segundos, 25.000 barras M5 no treino, lookback TCN de 32 barras, horizonte de uma barra e label `spot_forward`. O contrato executado dura 5 minutos. O label por fechamento é uma aproximação do desfecho do contrato e não representa o spot executado.

## Fluxo operacional

1. O WebSocket Deriv recebe ticks e velas; o ciclo abre a cada fronteira M5.
2. O checkpoint TCN local fornece a probabilidade direcional. CALL é escolhido quando a probabilidade calibrada é pelo menos 0,5; caso contrário, PUT.
3. O loss-classifier pode inverter o lado após aprendizado com contratos reais. O meta LightGBM fornece contexto adicional; filtros técnicos, de mercado e de cotação podem impedir a compra.
4. Kelly fracionário e recuperação calculam a stake. Sem checkpoint técnico compatível, nenhuma ordem é enviada. Com checkpoint válido, o teto inicial é **1% da banca por ordem**, inclusive em `cover_l0` e em DEMO ou REAL.
5. A proposta é revalidada antes da compra. O resultado do contrato é reconciliado com a Deriv e persistido.

O treino salva o checkpoint TCN local após verificar integridade e geometria. Não há qualificação estatística de deploy da TCN nem avaliação diagnóstica de liquidação por closes M5. A validação interna usada para ajuste e calibração não autoriza aumentar o teto inicial. A captura Timescale de ticks e contratos confirmados é independente da decisão de compra.

A política operacional mantém stop win por sessão e **não usa stop loss nem limite acumulado de perda**. O stop win configurado é 4,31% para a banca aplicável; o payout de referência é 0,85. O caminho de recuperação preserva seus limites próprios, sempre subordinados ao teto inicial de 1% quando o checkpoint local é usado.

Com o motor em execução, Prometheus coleta `/metrics` na porta 9100 e o Grafana consulta o Prometheus pelo datasource provisionado. A rota entre Docker no WSL e o motor no Windows está descrita em [observabilidade](docs/engineering-observability.md).

## Executar no WSL

Use o ambiente Conda `deriv-api` com Python 3.13.12. Configure `.env` a partir de `.env.example` com as credenciais da Deriv e da infraestrutura.

```bash
make app-install
make docker-up
python app/train.py
python app/scripts/operations/check_dl_checkpoint.py
python app/run.py
```

O launcher de treino em WSL é `app/scripts/wsl/launch-train-wsl.sh`. O motor usa o checkpoint produzido pelo treino separado (`online_training=false`). Para QA, execute `make app-pre-commit-run` no WSL; o pipeline inclui Vulture com confiança 100, lint, segurança, testes e cobertura de 100% em `app/src`.

## Documentação

- [Arquitetura e componentes](docs/arquitetura.md)
- [Treino e checkpoint TCN](docs/engineering-deep-learning.md)
- [Configuração SSOT](docs/engineering-settings-ssot.md)
- [Execução e risco](docs/binary-senior-playbook.md)
- [Infraestrutura Docker](docs/infra-docker.md)
- [Operação e regras para agentes](AGENTS.md)
- [CI/CD e workflows](.github/workflows/WORKFLOWS.md)
