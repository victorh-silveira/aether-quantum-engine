---
name: aether-v75-market-analyst
description: Analise quantitativa de indices sinteticos continuos 24/7 (Volatility 75 1s) e regimes macro.
---

# Análise de Mercado de Índices Sintéticos Contínuos

## 1. Contexto e Pré-requisitos
- Leia `docs/deriv-indices-algorithm.md` e `.agents/rules/aether-v75-market.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Compreensão da Dinâmica Contínua:**
   - Reconheca que indices sinteticos puros como `1HZ75V` operam ininterruptamente, sem gaps de sessao ou feriados, gerados por processos pseudoaleatorios da Deriv.
2. **Análise de Volatilidade e Regimes:**
   - Calcule bandas de Bollinger adaptativas, ATR relativo e coeficientes de Hurst para diagnosticar regimes de compressao ou expansao de volatilidade.
3. **Isolamento de Contratos:**
   - Mantenha a analise de indices sinteticos puramente para telemetria analitica, sem gerar compras operacionais Rise/Fall (as quais sao reservadas ao ativo `OTC_SSMI`).

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/domain/test_v75*.py -v
```

## 4. Critérios de Aceite
- Relatorio claro de metricas de volatilidade e regimes sem interferencia no caminho operacional ativo.
