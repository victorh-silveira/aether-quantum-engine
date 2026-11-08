---
name: aether-asyncio-supervisor
description: Supervisao de corotinas asyncio, prevencao de starvation do event loop, TaskGroup e shield.
---

# Supervisão e Concorrência Asyncio

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-python-313-runtime.md` e a regra `.agents/rules/aether-python-313-runtime.mdc`.
- O motor utiliza exclusivamente o loop assincrono padrao do CPython 3.13.

## 2. Procedimento Operacional Passo a Passo
1. **Detecção de Starvation do Loop:**
   - Verifique se funcoes marcadas com `async def` executam computacoes matematicas longas ou I/O de disco/socket sincrono sem `await`.
   - Mova rotinas de processamento pesado para `await asyncio.to_thread(funcao_sincrona, *args)`.
2. **Orquestração com TaskGroup:**
   - Substitua chamadas legadas de `asyncio.gather` por blocos `async with asyncio.TaskGroup() as tg:`.
   - Garanta que falhas em tarefas filhas cancelem as demais ordenadamente.
3. **Proteção de Encerramento (Graceful Shutdown):**
   - Proteja operacoes criticas de desconexao e persistencia de estado com `asyncio.shield()`.
   - Capture `asyncio.CancelledError`, execute a higienizacao de recursos e repropage o erro quando aplicavel.

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_aether_asyncio*.py -v
```

## 4. Critérios de Aceite
- Zero bloqueios na thread principal do loop.
- Tratamento estruturado de `CancelledError`.
- Testes assincronos executados sem corotinas orfas ou ResourceWarnings.
