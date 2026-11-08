---
name: aether-indicator
description: Desenvolvimento, revisao e execucao do servico de indicador publico de indices sintetico e OTC.
---

# Operação do Subsistema Indicador Público

## 1. Contexto e Pré-requisitos
- Leia `indicator/README.md`, `docs/strategy-pullback.md` e `.agents/rules/aether-indicator.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Isolamento de Processo:**
   - Confirme que o modulo `indicator/` consome unicamente endpoints publicos e nao possui imports de servicos autenticados de contratos.
2. **Execução de Diagnóstico:**
   - Execute o diagnostico de velas historicas:
     ```bash
     make indicator-train
     ```
3. **Auditoria de Testes e Cobertura:**
   - Execute a suite de testes especifica do indicador garantindo 100% de cobertura:
     ```bash
     make indicator-test
     ```

## 3. Comandos de Verificação
```bash
make indicator-lint
make indicator-test
```

## 4. Critérios de Aceite
- 100% de cobertura alcancada em `src/indicator/`.
- Painéis do Grafana exibindo leituras sem erros de consulta.
