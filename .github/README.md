# GitHub Actions

CI por stack real do Aether (Python, Docker, Shell). JSON e YAML nao sao stacks: validam-se em steps `Python | JSON *` e `Python | YAML *`. CD = semantic-release apos CI verde. Sem Kubernetes, Terraform, Azure, GHCR deploy ou destroy.

## Visao (push em `main`)

```mermaid
flowchart LR
  subgraph ci [CI paralelo]
    PY[Python]
    DK[Docker]
    SH[Shell]
  end
  PY --> R[Release]
  DK --> R
  SH --> R
  R --> S[Resumo]
```

| Fase | Job | Notas |
|------|-----|-------|
| CI | Python | Lint, JSON/YAML, Validate, Seguranca, Testes, Build |
| CI | Docker / Shell | Mesmo orquestrador e nomes do pre-commit |
| Release | Semantic release | Apos gates verdes no push `main` |

Crash-first em cada stack: lint, validate, security, test, build.

## Workflows

| Workflow | Gatilho | Uso |
|----------|---------|-----|
| [ci.yml](workflows/ci.yml) | push/PR `main`, manual | CI matriz; release no push `main` |

## Composite actions

```text
.github/actions/
├── shared/pipeline-summary/
└── ci/
    ├── setup-python/
    ├── validate-docker/
    ├── validate-shell/
    ├── release/
    └── sync-tags/
```

Pre-commit usa os mesmos nomes de step (`Python | Lint`, `Docker | Lint`, `Shell | Lint`).
