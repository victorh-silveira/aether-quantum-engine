---
name: aether-cycle-debug
description: Depuracao de ciclo do orquestrador, warmup de buffers, assinaturas de ciclo e watchdog.
---

# Depuração de Ciclos e Orquestração

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-orchestrator.md` e a regra `.agents/rules/aether-orchestrator.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Análise de Warmup:**
   - Inspecione se os buffers de candles M15 receberam o volume minimo configurado de barras historicas antes de transitar para o estado ACTIVE.
2. **Inspeção de Assinaturas de Ciclo:**
   - Monitore a geracao de assinaturas (epoch + symbol) para verificar se algum ciclo foi pulado ou executado em duplicidade.
3. **Depuração de EXEC_EMPTY:**
   - Se o motor emitir EXEC_EMPTY sistemico, rastreie qual dos 5 pilares do setup institucional ou gate de Killzone esta rejeitando a entrada.
4. **Auditoria de Watchdog:**
   - Inspecione se o watchdog esta registrando batimentos regulares (heartbeats) e se nao ha deadlocks assincronos.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_trading_cycle_entry*.py -v
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_watchdog_service.py -v
```

## 4. Critérios de Aceite
- Ciclos executando sincronizados no segundo zero de cada vela M15.
- Transicao clara de estados e justificativa transparente para cada SKIP.
