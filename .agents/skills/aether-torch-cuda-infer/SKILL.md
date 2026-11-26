---
name: aether-torch-cuda-infer
description: Inferencia e execucao assincrona com PyTorch, CUDA host e isolamento de recursos.
---

# Inferência PyTorch e Gestão de Recursos no Host

## 1. Contexto e Pré-requisitos
- Leia `docs/strategy-laya.md` e `.agents/rules/aether-laya-decision.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Configuração de Device:**
   - Instalar com `make app-cuda-install` e validar com `make app-cuda-check` e `make app-cuda-train-check`. Os alvos app-run e app-train exigem CUDA; o adapter auto isolado preserva fallback CPU.
   - O perfil de treino auto escolhe head em GPUs de ate 12 GiB. Exigir ao menos 3 GiB livres para head e 12 GiB livres para full; nao iniciar treino pesado em CPU.
2. **Execução em Modo Inferência:**
   - Envolva todas as predicoes em `with torch.inference_mode():`.
3. **Despacho Assíncrono:**
   - Nao execute `model(tensor)` diretamente no event loop. Despache com:
     ```python
     resultado = await asyncio.to_thread(model_predict, input_tensor)
     ```

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest tests/unit/infrastructure/test_laya_decision_adapter.py -v
```

## 4. Critérios de Aceite
- Predicoes sem vazamento de memoria GPU/VRAM.
- Event loop assincrono com latencia estavel.
