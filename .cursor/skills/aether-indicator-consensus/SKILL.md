---
name: aether-indicator-consensus
description: Integra sinais M5 aprovados do indicador com decisões TCN, meta e loss-classifier do motor automático.
---

# Consenso do indicador

1. Leia `AGENTS.md`, `indicator/AGENTS.md`, `docs/arquitetura.md` e `indicator/docs/modelos.md`.
2. Preserve a separação entre o cliente público do indicador e o cliente autenticado de contratos.
3. Exija mesma vela fechada, ativo, modelo aprovado/ativo e concordância entre lado original da TCN, lado final e indicador antes de montar uma ordem. M15/H1 permanecem observacionais; dados atrasados ou indisponíveis deixam o ativo sem ordem.
4. Teste ao menos dois índices no mesmo ciclo, divergência de lado, checkpoint ausente, erro de banco, saldo reservado e liquidação de contratos simultâneos.
5. Mostre no Grafana sinais observacionais aprovados e, separadamente, compras reais ainda ativas. Avalie versões combinadas apenas com liquidações confirmadas do broker.
6. Execute lint e testes do motor e do indicador; valide Compose, SQL e dashboards antes de integrar na `main`.
