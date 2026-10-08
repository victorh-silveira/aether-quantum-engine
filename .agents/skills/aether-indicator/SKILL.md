---
name: aether-indicator
description: Desenvolve e revisa o indicador publico de sinteticos, treino temporal, sinais sem contrato e dashboards Grafana.
---

# Indicador Aether

1. Leia `AGENTS.md`, `docs/arquitetura.md`, `docs/modelos.md` e `docs/operacao.md`.
2. Preserve a barreira do cliente público: apenas catálogo e candles; teste contra rotas de negociação.
3. Ao alterar sinais, teste fechamento de vela, lacunas, isolamento de ativo/período, modelo ausente e frescor.
4. Ao alterar treino, compare validação temporal com referência simples e registre versão por ativo/período.
5. Atualize Grafana, Prometheus, docs e regra junto com o código.
6. Rode `make lint`, `make test` e valide Compose/dashboard. Não afrouxe a cobertura de 100%.
7. Preserve a estrutura de jobs do CI anterior nesta branch e seus gatilhos próprios. A `main` contém o motor e CI/CD anteriores. Publicação GHCR do indicador somente após push aprovado.
