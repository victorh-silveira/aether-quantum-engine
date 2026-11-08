# Padrões de Engenharia de Software e Qualidade

## 1. Diretrizes Fundamentais de Qualidade

1. **Cobertura 100% Inegociável:** Todo código sob `src/` e `src/indicator/` deve manter 100% de cobertura de linhas e ramos. Reduções de cobertura resultam em falha imediata no CI.
2. **Limite Estrutural de 300 Linhas:** Nenhum arquivo Python em `src/**/*.py` pode ultrapassar 300 linhas de código.
3. **Pipeline de Linters Estrito (Crash-First):**
   - **Ruff:** Verificação sintática e formatação automática.
   - **Interrogate:** 100% de docstrings em módulos, classes e funções.
   - **Vulture:** Detecção e eliminação de código morto.
   - **Bandit & Gitleaks:** Auditoria de segurança e prevenção de vazamento de segredos.
4. **Commits Padronizados em PT-BR:**
   - Mensagens estritamente em Português no formato Conventional Commits.
   - Execução via `scripts/wsl/commit.sh` ou `make app-commit`.

---

## 2. Comandos Oficiais de Validação

```bash
# Validacao completa de linters e docstrings
make app-lint

# Suite completa de testes unitarios com cobertura 100%
make app-test

# Testes do subsistema indicador
make indicator-test

# Validacao da matriz de cobertura de agentes
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/repo/test_agent_coverage_matrix.py -v
```
