# Prompt-modelo de engenharia

Contrato reutilizavel derivado deste repositorio de referencia (**Aether Quantum Engine**). Use este arquivo para orientar um agente a gerar ou adaptar qualquer projeto (Python ou outra linguagem) com o mesmo padrao de engenharia: DDD, hexagonal, TDD, qualidade, DX e documentacao.

Dominios de negocio de outros PoCs (ex.: OTRS → Google Chat) sao apenas exemplos historicos de aplicacao do padrao — **nao** regra obrigatoria do novo projeto. O negocio deste repo (Deriv / R_10 / TCN) tambem nao deve ser copiado cegamente.

## 1. Papel e objetivo

Voce e um engenheiro senior. Ao receber um novo dominio de negocio e (opcionalmente) uma linguagem, deve:

1. Projetar e implementar a solucao no padrao deste contrato.
2. Adaptar ferramentas e pastas a linguagem escolhida sem abandonar os invariantes.
3. Manter docs, Makefile, hooks e gates alinhados ao codigo.
4. Preferir TDD: testes primeiro nas camadas de dominio e application; integracao nos adapters de IO.

Objetivo final: o novo repositorio deve “parecer” este projeto em arquitetura, qualidade e operacao — nao copiar o dominio Aether nem outro PoC.

## 2. Contrato invariavel (language-agnostic)

### Camadas (hexagonal / DDD)

| Camada | Conteudo | Pode depender de |
|--------|----------|------------------|
| domain | Entidades, value objects, domain services, validacoes | Ninguem de fora do dominio |
| application | Use cases + ports (contratos) | domain |
| infrastructure | Adapters (HTTP, DB, filas), config, logging de borda | application, domain |
| presentation | CLI/API/UI + composition root | application, domain, infrastructure |

Regras:

- `domain` e `application` nao importam `infrastructure` nem `presentation`.
- Ports sao contratos (Protocol / interface / trait). Adapters implementam ports na `infrastructure`.
- Validacao de entidades fica no dominio.
- Formatacao de payload / regras de saida ficam no dominio ou application — nao na CLI/controller.
- Domain e use case nao emitem logs. Logging semantico fica na presentation e/ou adapters.
- Composition root (wiring Settings → adapters → use case) fica na presentation.
- Neste repo de referencia: motor **Python 3.13 + asyncio no host**; sidecars Docker; DataFrame **Polars-only**; inferencia critica CUDA no host; event loop sem bloqueio de CUDA/Polars pesado (offload). Doutrina: `docs/engineering-architecture-senior.md`.

### Qualidade e estilo

- Sem comentarios no codigo de aplicacao (codigo autodescritivo; docstrings OK se a linguagem/projeto exigir).
- Documentacao tecnica em PT-BR; sem emojis em codigo, logs ou docs.
- Conventional Commits (assunto PT-BR neste repo).
- Segredos nunca no git: `.env` na raiz + `.env.example`; neste repo knobs operacionais tambem vivem em `config/settings.json` com `resolve_*` (SSOT).
- Cobertura alta nas camadas de app (neste repo: **100%** em `app/src`).
- Limite de tamanho por arquivo de codigo (~**300** linhas); extrair quando passar.
- Type checking estrito quando a linguagem permitir.

### Testes (TDD)

- Unitarios por camada: `tests/unit/{domain,application,infrastructure,presentation}` (espelhar pastas reais do projeto).
- Integracao para adapters de IO (HTTP mock, DB fake/testcontainer, etc.).
- Application: preferir fakes dos ports.
- Isolar testes do `.env` local (flag/env do tipo `*_DISABLE_DOTENV` quando aplicavel).

## 3. Arquitetura de pastas (template)

Ajuste nomes de entrypoint/build file a linguagem; preserve a intencao das pastas.

```text
.
├── app/
│   ├── src/
│   │   ├── domain/
│   │   ├── application/
│   │   ├── infrastructure/
│   │   └── presentation/
│   ├── tests/
│   │   ├── unit/
│   │   │   ├── domain/
│   │   │   ├── application/
│   │   │   ├── infrastructure/
│   │   │   └── presentation/
│   │   └── integration/
│   ├── scripts/
│   │   └── operations/       # orquestrador de lint/test/security/clean
│   ├── pyproject.toml        # ou Cargo.toml / go.mod / package.json / etc.
│   ├── requirements.txt      # se aplicavel
│   └── requirements-dev.txt
├── docs/
│   ├── arquitetura.md
│   ├── structure.md
│   ├── engineering-standards.md   # QA / gates (neste repo)
│   ├── engineering-<lang>-deps.md # se houver politica de deps
│   ├── engineering-observability.md
│   └── CHANGELOG.md
├── infra/                    # so se houver stack local / integracao
├── linters/
│   ├── commitlint.config.mjs
│   ├── git-hooks/
│   └── releaserc.json        # se usar semantic-release
├── config/                   # settings operacionais (neste repo: settings.json)
├── .cursor/                  # rules + skills versionadas (superficie agentes)
├── Makefile
├── AGENTS.md
├── prompt-model.md
├── README.md
├── .env.example
└── run.py                    # ou equivalente na raiz
```

Imports da aplicacao devem ser limpos (sem prefixo desnecessario tipo `app.src`): `from domain...`, `from application...` (ou o layout idiomatico equivalente com a mesma regra de dependencia).

## 4. Qualidade e DX

### Gates conceituais (obrigatorios em qualquer linguagem)

| Gate | Intencao |
|------|----------|
| Lint / format | Estilo consistente, imports ordenados |
| Type check | Tipos estritos (quando a linguagem permitir) |
| Dead code | Eliminar codigo morto |
| Test + coverage | Unit + integracao; fail-under alto |
| Security | SAST + auditoria de dependencias (+ secrets scan no CI) |
| Hooks | pre-commit + commit-msg (Conventional Commits) |
| Make | `app-install`, `app-lint`, `app-test`, `app-security`, `app-clean`, `app-setup` |

Orquestrador central (neste repo: `app/scripts/operations/clean_workspace.py`) deve ser o unico ponto chamado pelo Makefile para lint/test/security/clean.

### Mapa de ferramentas (referencia → equivalentes)

| Conceito | Python (este repo) | Exemplos em outras linguagens |
|----------|--------------------|-------------------------------|
| Format/lint | Ruff (+ Interrogate) | gofmt+golangci-lint, rustfmt+clippy, eslint+prettier |
| Types | tipagem forte onde couber | tsc strict, generics + compiler |
| Dead code | Vulture | deadcode / knip / warnings do compilador |
| Test | pytest + xdist | go test, cargo test, jest/vitest, JUnit |
| Coverage | pytest-cov fail-under **100%** em `src` | go cover, tarpaulin, istanbul — manter barra alta |
| Security | Bandit + pip-audit | gosec/govulncheck, cargo-audit, npm audit, trivy |
| Hooks | pre-commit + commitlint | husky + commitlint, lefthook |
| Release | semantic-release | mesma ideia SemVer + changelog |

Para Python novo: preserve o stack deste projeto (Ruff, Interrogate, Vulture, pytest-cov 100%, Bandit, pip-audit, pre-commit, commitlint, Makefile). DataFrame SSOT neste repo = **Polars only** (ver `docs/engineering-python-deps.md`).

## 5. Logging semantico

Principios (language-agnostic):

- Eventos nomeados e estaveis (`*.started|finished|failed|skipped_*` ou tags de ciclo equivalentes).
- Caminho feliz: poucas linhas INFO (~3 no fluxo curto; neste repo o ciclo usa tags densas mas dedupe/quiet channels).
- Sem dump de payload JSON nem body HTTP em INFO.
- Redact de URLs/segredos; nunca secret cru em log.
- `exc_info` / stack apenas em DEBUG nas falhas (salvo politica explicita).
- Silenciar loggers ruidosos de HTTP clients em WARNING+.

Neste repo: `get_logger` / `setup_logger`, tags de ciclo (CLUSTER → SCALE → FUSION → …), inventario em `docs/engineering-observability.md` e `docs/engineering-logging-inventory.md`. Nao forcar API de outro PoC; preserve o principio.

## 6. Documentacao obrigatoria

Manter alinhada ao codigo:

| Doc | Papel |
|-----|-------|
| `docs/arquitetura.md` | Camadas, ports, fluxo, config |
| `docs/structure.md` | Arvore e regras de dependencia |
| `docs/engineering-standards.md` | Qualidade, testes, entrypoints / gates |
| `docs/engineering-python-deps.md` | Politica de deps (se Python) |
| `docs/engineering-observability.md` | Logging / anti-poluicao |
| `docs/infra-*.md` | Se houver infra local |
| `AGENTS.md` | Prioridades curtas para agentes |
| `docs/agent-coverage.md` | Matriz doc + rule + skill (neste repo) |
| `README.md` | Setup, Make, badges/CI se houver |
| `.env.example` | Vars sem segredos reais |
| `prompt-model.md` | Este contrato (raiz) |

## 7. Superficie de agentes (Cursor)

Neste repositorio a superficie e versionada:

- Rules: `.cursor/rules/*.mdc` (`alwaysApply: true`)
- Skills: `.cursor/skills/*/SKILL.md`
- Indices: `AGENTS.md` + `docs/agent-coverage.md`
- Fechamento de mudanca: skill `aether-surface-sync` + `docs/engineering-surface-sync.md` (sync docs/rules/skills, pre-commit WSL, anti-sujeira)

Em projeto novo: ou adote a mesma matriz, ou documente equivalente minimo em `AGENTS.md`.

## 8. Protocolo de adaptacao (checklist do agente)

1. Ler o dominio/requisitos do novo projeto e a linguagem alvo.
2. Listar entidades, use cases, ports e adapters necessarios.
3. Criar a arvore de pastas e impor regras de dependencia (CI ou lint de imports se possivel).
4. Configurar build, `.env.example`, Makefile e orquestrador de qualidade; se houver knobs, SSOT tipo `settings.json` + `resolve_*`.
5. Instalar hooks (pre-commit + Conventional Commits).
6. Implementar em TDD: domain → application (fakes) → adapters → presentation.
7. Adicionar `infra/` somente se o contexto exigir runtime local; nao copiar stack Aether/Docker cegamente.
8. Escrever/atualizar docs da secao 6 e `AGENTS.md` (e matriz se existir).
9. Rodar `make app-lint`, `make app-test`, `make app-security` (ou equivalentes) ate verde.
10. Garantir commits Conventional Commits e ausencia de segredos; fechar com surface sync se houver superficie Cursor.

## 9. Anti-padroes

- HTTP client, SQL ou framework web dentro de domain / application.
- Comentarios no codigo de aplicacao.
- Logs dentro de use case ou entidade.
- Composition root espalhado ou fora de presentation.
- Cobertura “quase” o suficiente; baixar fail-under sem decisao explicita.
- Commits fora de Conventional Commits.
- Docs / rules / skills desatualizadas apos mudanca de arquitetura.
- Commitar `.env`, tokens ou artefactos `_tmp*`.
- Copiar dominio Deriv/TCN/OTRS/Docker sem necessidade do novo dominio.

## 10. PROMPT PARA COLAR

Copie o bloco abaixo integralmente em um novo chat/projeto. Substitua apenas as linhas marcadas com `<<< >>>`.

```text
Voce e um engenheiro senior. Adapte ou crie o repositorio abaixo para ficar equivalente ao contrato de engenharia do projeto de referencia (DDD + hexagonal + TDD + DX forte). Contrato: prompt-model.md do repositorio Aether Quantum Engine.

NOVO PROJETO
- Nome: <<<NOME>>>
- Dominio / objetivo: <<<DESCRICAO DO NEGOCIO>>>
- Linguagem/runtime: <<<python|go|rust|ts|outra>>>
- Entrada principal: <<<CLI|API HTTP|worker|outro>>>
- Integracoes externas: <<<listar ou "nenhuma ainda">>>
- Infra local necessaria: <<<sim/nao; se sim, o minimo>>>

INVARIANTES (obrigatorios)
1. Camadas: domain, application (ports + use_cases), infrastructure (adapters/config/logging), presentation (composition root).
2. domain e application NAO importam infrastructure nem presentation.
3. Ports como contratos; adapters na infrastructure.
4. Validacao no dominio; formatacao de payload em domain/application, nao na borda de UI/CLI.
5. Domain e use case NAO logam; logging semantico na presentation/adapters (eventos estaveis; sem dump de payload/segredos; redact de URLs; caminho feliz enxuto em INFO).
6. Sem comentarios no codigo; docs em PT-BR; sem emojis; Conventional Commits.
7. Config via .env na raiz + .env.example; nunca commitar segredos; knobs operacionais com SSOT explicito se existirem.
8. Testes TDD: unit por camada + integracao nos adapters de IO; fakes nos ports da application.
9. Cobertura alta nas camadas de app (meta 100% se a ferramenta permitir).
10. Gates via Makefile: install, lint, test, security, clean, setup; orquestrador unico em app/scripts/operations (ou equivalente).
11. Limite ~300 linhas por arquivo de codigo.
12. Type checking estrito quando a linguagem permitir.

PASTAS ALVO
app/src/{domain,application,infrastructure,presentation}
app/tests/{unit,integration}
app/scripts/operations/
docs/{arquitetura,structure,engineering-standards,engineering-observability}.md
linters/ (commitlint + git-hooks)
Makefile, AGENTS.md, prompt-model.md, README.md, .env.example, entrypoint na raiz
infra/ somente se necessario ao dominio
.cursor/ rules+skills se adotar superficie de agentes

QUALIDADE
- Python: Ruff, Interrogate, Vulture, pytest+cov fail-under 100, Bandit, pip-audit, pre-commit, commitlint; DataFrame SSOT = Polars se houver tabelas.
- Outra linguagem: equivalentes idiomaticos mantendo os MESMOS gates conceituais.

DOCS A MANTER ALINHADAS AO CODIGO
docs/arquitetura.md, docs/structure.md, docs/engineering-standards.md, docs/engineering-observability.md, AGENTS.md, README.md, prompt-model.md

PROTOCOLO
1) Mapear entidades/use cases/ports/adapters do dominio informado.
2) Criar arvore e regras de dependencia.
3) Configurar build, env, Makefile, hooks.
4) Implementar TDD (domain → application → adapters → presentation).
5) Nao copiar stack Aether/Docker/OTRS a menos que o dominio peca algo equivalente.
6) Atualizar docs e AGENTS.md (e agent-coverage se existir).
7) Rodar gates ate verde; fechar com surface sync se houver .cursor/.

ANTI-PADROES
HTTP/DB no dominio; comentarios; logs no use case; composition root fora de presentation; commits nao convencionais; docs defasadas; segredos no git.

Entregue: estrutura, codigo, testes, Makefile, docs e AGENTS.md coerentes com este contrato.
```

## Referencia deste repositorio

Projeto que originou/materializa o contrato (padrao de engenharia, nao template de negocio):

- [`AGENTS.md`](AGENTS.md)
- [`docs/arquitetura.md`](docs/arquitetura.md)
- [`docs/structure.md`](docs/structure.md)
- [`docs/engineering-standards.md`](docs/engineering-standards.md)
- [`docs/engineering-python-deps.md`](docs/engineering-python-deps.md)
- [`docs/engineering-observability.md`](docs/engineering-observability.md)
- [`docs/engineering-logging-inventory.md`](docs/engineering-logging-inventory.md)
- [`docs/engineering-surface-sync.md`](docs/engineering-surface-sync.md)
- [`docs/agent-coverage.md`](docs/agent-coverage.md)
- [`docs/infra-docker.md`](docs/infra-docker.md)
- [`Makefile`](Makefile)
- [`app/scripts/operations/clean_workspace.py`](app/scripts/operations/clean_workspace.py)
