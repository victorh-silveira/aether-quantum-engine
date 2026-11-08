---
name: aether-settlement-debug
description: Depuracao de liquidacao de contratos, fila Redis ZSET, reconciliacao e profit_table.
---

# Depuração de Liquidação e Reconciliação

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-settlement.md` e `.agents/rules/aether-settlement.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Inspeção da Fila Redis:**
   - Inspecione a chave ZSET `settlement:queue:priority` para identificar contratos que estejam com vencimento atrasado.
2. **Reconciliação Forçada via REST:**
   - Em caso de falha no stream WebSocket, consulte o endpoint `profit_table` da Deriv para obter o status definitivo do contrato.
3. **Resolução de Contratos Órfãos:**
   - Identifique contratos sem resposta e aplique a quarentena de 900s no ativo afetado para proteger o saldo contra novas compras antes da solucao.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_settlement*.py -v
```

## 4. Critérios de Aceite
- Fila de liquidacao limpa e sem contratos travados.
- Conciliacao contabil 100% precisa com o extrato da corretora.
