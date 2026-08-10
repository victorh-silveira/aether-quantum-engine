# AGENTS.md — Aether Quantum Engine

Ponto de entrada para agentes Cursor/LLM neste repositorio.

## Idioma e ambiente

- Respostas e commits em **PT-BR**
- Terminal/scripts: **WSL Linux** (nunca CMD/PowerShell nativo)
- Sem emojis em codigo, logs ou docs tecnicos
- Sem comentarios no codigo (docstrings OK)

## Universo operacional

- Simbolo unico: **`R_10`** (Volatility 10 / Deriv) — **M2**
- Relogio: contrato **2 m**; micro/MINI **120 s**; macro **3600 s**; ciclo **120 s**
- SSOT: `config/settings.json` + `app/src/domain/symbols/drift_symbols.py`
- Artefactos/treino antigos (OTC_SPC/M15 ou gran 60/300/900) sao invalidos apos migracao para **R_10** M2

## O que o LLM e / nao e

- **E:** copiloto de engenharia e auditoria
- **Nao e:** decisor de CALL/PUT em runtime (TCN + meta + gates/Kelly)

Doutrina: [`docs/llm-trading-doctrine.md`](docs/llm-trading-doctrine.md)  
Matriz 100% cobertura: [`docs/agent-coverage.md`](docs/agent-coverage.md)  
Rules/skills versionadas: [`.cursor/rules/`](.cursor/rules/) e [`.cursor/skills/`](.cursor/skills/)

## Proibicoes globais

- `force_trade_every_cycle=true` como “fix” de EXEC_EMPTY
- Revenge sizing apos LOSS; “operar mais para aprender” com N baixo
- Remover caps de stake, fila de settlement ou timeouts “temporariamente”
- Arquivos `app/src/**/*.py` acima de **300 linhas**
- Cobertura de testes em `app/src` abaixo de **100%**
- Assunto de commit em ingles; escopo fora do enum commitlint

Nota operacional (**escopo 1.1** + arquitetura continua R_10): quality gate amplo (RSI/price_zone/SIDE_EQ block) permanece **fora**; `signal_skip` mini/cal = soft Kelly; **loss-clf** soft na faixa media (`veto_ready`) e **FLIP** CALL↔PUT **relativo ao TCN** se `p_loss >= hard_p_loss_floor` (**0.90**, so com `veto_ready`; seed bootstrap devolve **p_loss real**, sem COLD_START neutro); **nunca FLIP** se Edge Cal do TCN >= floor (`FLIP_BLOCK:tcn_edge`); `p_loss >= flip_waive_scale_above_p_loss` (**0.95**) override seed/SCALE (`why=p_ovr`) + soft neg_edge; vela no alvo usa `flip_candle_p_loss_floor` (**0.85**) so se TCN fraco; soft candle/p_ovr so se edge >= **−0.12**; **chop** = soft Kelly (**0.55**); **neg_edge** = hard-skip salvo excecoes candle/p_ovr acima. Sem revenge sizing pos-LOSS. Vies CALL/PUT: treino + SIDE_EQ soft.

## Escopos commitlint

`all`, `api`, `app`, `config`, `deps`, `domain`, `engine`, `infra`, `llm`, `orchestrator`, `pres`, `release`, `repo`, `risk`, `scripts`, `test`, `tools`, `ws`

Formato: `tipo(escopo): assunto em PT-BR` + corpo obrigatorio.

## Pre-commit

`.pre-commit-config.yaml` → `clean_workspace.py` stages: lint, test (cov 100%), security, cleanup; commit-msg: commitlint.

## Leitura por tarefa

| Tarefa | Abrir primeiro |
|--------|----------------|
| Qualquer mudanca | este arquivo + `docs/agent-coverage.md` |
| CALL/PUT/SKIP senior | `docs/binary-senior-playbook.md` + skill `aether-binary-senior` |
| Loss-classifier / Docker ml | `docs/infra-docker.md` + skill `aether-infra-stack` |
| Risco / logs de sessao | doutrina + skill `aether-session-review` |
| Knob em settings | `docs/engineering-settings-ssot.md` + skill `aether-settings-change` |
| Ciclo / warmup | `docs/engineering-orchestrator.md` + skill `aether-cycle-debug` |
| Scale vision / raw_extreme | `docs/engineering-orchestrator.md` + playbook + skills `aether-cycle-debug` / `aether-binary-senior` |
| Settlement | `docs/engineering-settlement.md` + skill `aether-settlement-debug` |
| DL / treino / vies de classe | `docs/engineering-deep-learning.md` + skill `aether-dl-train` |
| Docker / Redis | `docs/infra-docker.md` + skill `aether-infra-stack` |
| Deriv PAT/WS | `docs/deriv-api-aether.md` + skill `aether-deriv-connect` |
| QA / pre-commit | `docs/engineering-standards.md` + skill `aether-precommit` |

Inventario de modulos: [`docs/structure.md`](docs/structure.md)  
Arquitetura runtime: [`docs/arquitetura.md`](docs/arquitetura.md)
