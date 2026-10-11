# AGENTS.md — Aether Synthetic Indicator

- Responder e documentar em PT-BR. Trabalhar no WSL com Python 3.13.
- Produto: observação dos índices Volatility públicos da Deriv em M5, M15 e H1. Forex é operação manual externa; não há inferência cruzada.
- Serviço ativo no Compose: `app/src/indicator/`, entrada `app/run.py`, treino `app/train.py`, configuração `config/settings.json`.
- Selecionar somente `R_<número>` e `1HZ<número>V` ativos no catálogo sintético. Somente `active_symbols` e `ticks_history` no cliente público. Nunca incluir `buy`, `proposal`, `authorize`, liquidação, Kelly, recuperação ou credenciais de conta.
- `SEM SINAL` é obrigatório sem histórico contínuo, dado atual ou modelo próprio. Não reutilizar o último lado. Distinguir sempre `APROVADO` (supera maioria em acurácia e Brier) de `EXPERIMENTAL` (modelo avaliado sem ganho nos dois critérios); este último nunca deve ser exibido como sinal CALL/PUT ativo nos dashboards.
- Modelo por ativo e período, validação temporal contra maioria em acurácia e Brier. Histórico observado é direção da vela seguinte, não resultado de contrato.
- A previsão confirmada usa apenas velas fechadas e não é reescrita dentro do período. Preço e direção da vela em andamento podem ser exibidos como observação provisória, com timestamp e sem alegar acerto antecipado.
- Relatar métricas da validação por par; arquivar checkpoint anterior se uma nova avaliação reprovar esse par.
- Domínio puro; I/O em adapters; nenhuma interpolação de dados. Arquivos Python em `app/src` com no máximo 300 linhas; 100% de cobertura de linhas.
- Grafana usa `localhost:3001` e credencial `admin`/`aether` nas duas branches; o Grafana do motor usa `localhost:3000`.
- Os dashboards de sinais mostram apenas pares com última leitura CALL/PUT válida e atual; o dashboard de catálogo lista todos os índices ativos, inclusive os sem sinal.
- Os dashboards de sinais exigem versão do modelo ainda ativa em `indicator_model_versions`; `candle_epoch` representa abertura da vela, e o fechamento previsto ocorre em `candle_epoch + 2 * período`.
- Atualizar código, testes, documentação, dashboards, regras e skills em conjunto. Executar `make lint` e `make test`. Preservar a interface `app-*`/`docker-*` da `main` no Makefile, com `make help` como alvo padrão.
- Na `main`, este diretório permanece público e observacional. O motor autenticado em `../app/` consulta apenas sinais M5 aprovados para exigir concordância com a direção final; não introduzir compras neste subsistema. O CI principal valida este código antes de publicar o motor.

Ver [arquitetura](docs/arquitetura.md), [operação](docs/operacao.md), [modelos](docs/modelos.md) e [matriz de agentes](docs/agent-coverage.md).
