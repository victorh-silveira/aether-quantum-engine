# AGENTS.md — Aether Synthetic Indicator

- Responder e documentar em PT-BR. Trabalhar no WSL com Python 3.13.
- Produto: observação dos mercados sintéticos públicos da Deriv em M5, M15 e H1. Forex é operação manual externa; não há inferência cruzada.
- Serviço ativo no Compose: `app/src/indicator/`, entrada `app/run.py`, treino `app/train.py`, configuração `config/settings.json`.
- Somente `active_symbols` e `ticks_history` no cliente público. Nunca incluir `buy`, `proposal`, `authorize`, liquidação, Kelly, recuperação ou credenciais de conta.
- `SEM SINAL` é obrigatório sem histórico contínuo, dado atual ou modelo próprio aprovado. Não reutilizar o último lado.
- Modelo por ativo e período, validação temporal contra maioria em acurácia e Brier. Histórico observado é direção da vela seguinte, não resultado de contrato.
- Domínio puro; I/O em adapters; nenhuma interpolação de dados. Arquivos Python em `app/src` com no máximo 300 linhas; 100% de cobertura de linhas.
- Atualizar código, testes, documentação, dashboards, regras e skills em conjunto. Executar `make lint` e `make test`.
- `main` mantém o motor anterior e seu CI/CD original. `codex/synthetic-indicator` mantém o indicador com a mesma estrutura de jobs e validações do CI anterior, adaptada ao código presente. Push e PR rodam na respectiva branch; imagem GHCR do indicador somente após push aprovado. Sem release automático do indicador.

Ver [arquitetura](docs/arquitetura.md), [operação](docs/operacao.md), [modelos](docs/modelos.md) e [matriz de agentes](docs/agent-coverage.md).
