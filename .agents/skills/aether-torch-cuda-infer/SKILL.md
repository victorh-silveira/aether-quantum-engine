---
name: aether-torch-cuda-infer
description: Inferencia e execucao assincrona com PyTorch, CUDA host e isolamento de recursos.
---

# Inferência PyTorch e Gestão de Recursos no Host

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-deep-learning.md` e `.agents/rules/aether-deep-learning.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Configuração de Device:**
   - Detecte a disponibilidade de GPU via `torch.cuda.is_available()`. Se ausente, faca fallback gracioso para CPU sem falhas.
2. **Execução em Modo Inferência:**
   - Envolva todas as predicoes em `with torch.inference_mode():`.
3. **Despacho Assíncrono:**
   - Nao execute `model(tensor)` diretamente no event loop. Despache com:
     ```python
     resultado = await asyncio.to_thread(model_predict, input_tensor)
     ```

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/infrastructure/test_torch*.py -v
```

## 4. Critérios de Aceite
- Predicoes sem vazamento de memoria GPU/VRAM.
- Event loop assincrono com latencia estavel.
