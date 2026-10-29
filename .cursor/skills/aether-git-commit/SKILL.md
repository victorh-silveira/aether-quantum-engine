---
name: aether-git-commit
description: >-
  Padrao e executor obrigatorio de commits Git no Aether via script commit.sh,
  assegurando 100% de conformidade com os hooks de pre-commit e commitlint (Conventional Commits PT-BR).
  Use when committing changes, preparing commit messages, diagnosing commit-msg hook failures,
  or when the user asks to commit code or updates.
---

# Git Commit Padronizado Aether

## Visao Geral

No repositorio Aether Quantum Engine, os commits sao estritamente auditados por hooks locais (`linters/git-hooks/pre-commit` e `linters/git-hooks/commit-msg`) e pelo CI.
Para evitar falhas de validacao tardia (apos bateria de linters e testes demorados), todo commit deve ser disparado atraves do script padronizado:
`app/scripts/wsl/commit.sh` (ou atalho `tools/commit.sh` / `make app-commit`).

## Como Executar

### 1. Formato Posicional Direto
```bash
bash app/scripts/wsl/commit.sh <tipo> <escopo> "<assunto em PT-BR>" "<corpo descritivo detalhado>"
```

### 2. Com Stage Automatico (`-a`)
```bash
bash app/scripts/wsl/commit.sh -a -t <tipo> -s <escopo> -m "<assunto em PT-BR>" -b "<corpo descritivo detalhado>"
```

### 3. Simulacao sem Commitar (`--dry-run`)
```bash
bash app/scripts/wsl/commit.sh <tipo> <escopo> "<assunto>" "<corpo>" --dry-run
```

### 4. Via Makefile
```bash
make app-commit ARGS='<tipo> <escopo> "<assunto>" "<corpo>"'
```

## Regras Obrigatorias do Commitlint

### Tipos Permitidos (`type-enum`)
- `build`: Alteracoes no sistema de build ou dependencias externas
- `chore`: Tarefas de manutencao que nao alteram codigo de producao ou testes
- `ci`: Mudancas em arquivos e scripts de configuracao de CI/CD
- `docs`: Somente mudancas na documentacao
- `feat`: Uma nova funcionalidade (feature)
- `fix`: Correcao de um erro (bug fix)
- `perf`: Mudanca de codigo focada em melhoria de performance
- `qa`: Relacionado a Garantia de Qualidade/Processos de homologacao
- `refactor`: Mudanca de codigo que nao corrige bug nem adiciona funcionalidade
- `revert`: Reversao de um commit anterior
- `style`: Mudancas que nao afetam o sentido do codigo (espacos, formatacao)
- `test`: Adicao ou correcao de testes existentes

### Escopos Permitidos (`scope-enum`)
`all`, `api`, `app`, `config`, `deps`, `domain`, `engine`, `infra`, `llm`, `orchestrator`, `pres`, `release`, `repo`, `risk`, `scripts`, `test`, `tools`, `ws`

### Idioma PT-BR no Assunto
O assunto do commit deve ser escrito em Portugues (PT-BR). Palavras em ingles como `add`, `update`, `remove`, `fix`, `test`, `clean`, `change`, `implement`, etc., sao estritamente proibidas e causam rejeicao imediata.

### Corpo Obrigatorio
- O corpo descritivo e OBRIGATORIO (`body-empty: never`).
- Deve haver uma linha em branco entre o assunto e o corpo (`body-leading-blank: always`).
- O header (tipo(escopo): assunto) nao pode ultrapassar 100 caracteres.

## Docs e Relacionados
- Script: `app/scripts/wsl/commit.sh` e `tools/commit.sh`
- Regras: `linters/commitlint.config.mjs` e `.cursor/rules/aether-git-commit.mdc`
- Hook: `linters/git-hooks/commit-msg`
