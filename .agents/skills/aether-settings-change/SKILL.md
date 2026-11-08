---
name: aether-settings-change
description: Procedimento formal para alteracao de configuracoes e knobs em config/settings.json.
---

# Alteração Controlada de Configurações (Settings SSOT)

## 1. Contexto e Pré-requisitos
- Leia `docs/engineering-settings-ssot.md` e `.agents/rules/aether-settings-ssot.mdc`.

## 2. Procedimento Operacional Passo a Passo
1. **Planejamento da Mudança:**
   - Identifique a chave exata em `config/settings.json` a ser alterada e formule a hipotese tecnica para a mudanca.
2. **Edição e Validação de Esquema:**
   - Modifique o valor assegurando conformidade de tipo (int, float, bool, str) e validacao de limites defensivos.
3. **Execução de Testes de Doutrina:**
   - Execute o teste congelado de invariantes para garantir que limites soberanos de protecao nao foram afrouxados:
     ```bash
     /mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_doctrine_settings_ssot.py -v
     ```

## 3. Comandos de Verificação
```bash
/mnt/c/Users/victor-silveira/anaconda3/envs/deriv-api/python.exe -m pytest app/tests/unit/application/test_doctrine*.py -v
```

## 4. Critérios de Aceite
- `settings.json` validado com JSON lint.
- Testes de invariantes de doutrina passando com 100% de sucesso.
