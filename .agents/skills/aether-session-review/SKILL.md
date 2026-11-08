---
name: aether-session-review
description: Revisao de sessoes de trading, auditoria de logs, dimensionamento de risco e payouts.
---

# Revisão de Sessões de Trading e Auditoria

## 1. Contexto e Pré-requisitos
- Leia `docs/llm-trading-doctrine.md`, `docs/medallion.md` e `.agents/rules/aether-risk-sizing.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Auditoria de Execuções:**
   - Inspecione os logs da sessao procurando por anomalias de conexao, rejeicoes de proposta ou violacoes de horario de Killzone.
2. **Auditoria de Sizing e Payout:**
   - Verifique se cada ordem respeitou estritamente a stake base de 1% e o teto de 5% na recuperacao.
   - Confirme se todas as ordens em REAL possuíam EV positivo comprovado no momento da cotacao.
3. **Balanço da Sessão:**
   - Compare os resultados obtidos na `profit_table` da corretora com o estado local e as gravacoes no TimescaleDB.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe scripts/operations/check_dl_checkpoint.py
```

## 4. Critérios de Aceite
- Relatorio de sessao consistente, sem discrepancias de saldo ou execucoes fora de parametro.
