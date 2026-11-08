# Playbook de Execução Sênior: Opções Binárias Rise/Fall M15

## 1. Princípios Operacionais Inegociáveis

1. **Execução Baseada em Evidência:** Decisões de mercado são governadas por confluência matemática determinística.
2. **Ativo Âncora Oficial:** `OTC_SSMI` (Swiss Market Index 20 / Deriv OTC).
3. **Cadência Temporal:** Timeframe M15 fixo (900 segundos). Entrada autorizada exclusivamente nos primeiros 60 segundos da vela.
4. **Respeito aos Hard Gates:** Se qualquer um dos 5 pilares falhar, a ação imediata é **SKIP** (não operar).

---

## 2. Checklist de Validação do Sinal M15

Antes de disparar a ordem, o motor valida a seguinte árvore de decisão:

```
[1] KILLZONE BRT ATIVA?
    ├── 03h-08h | 10h-13h | 14h-16h BRT
    └── SE NÃO ──> VETO (SKIP: Fora da Killzone)

[2] TENDÊNCIA ESTRUTURAL M15 ALINHADA?
    ├── CALL: EMA20 > EMA50 (ascendentes)
    ├── PUT:  EMA20 < EMA50 (descendentes)
    └── SE NÃO ──> VETO (SKIP: Mercado sem Tendencia)

[3] RETRAÇÃO DE VOLATILIDADE SAUDÁVEL?
    ├── Recuo anterior <= 0.5 ATR14 da EMA20
    ├── Sem fechar além da EMA50
    └── SE NÃO ──> VETO (SKIP: Recuo Fora do Padrao)

[4] PREÇO DA SESSÃO UTC DENTRO DE BANDAS?
    ├── Fechamento favorável à média de preço típico
    ├── Contido em +/- 1.5 desvios padrão da sessão
    └── SE NÃO ──> VETO (SKIP: Exaustao Estatistica)

[5] FAIR VALUE GAP MITIGADO?
    ├── Imbalance entre t-2 e t testado pelo recuo
    └── SE NÃO ──> VETO (SKIP: Desequilibrio Nao Mitigado)

[6] JANELA TEMPORAL VÁLIDA?
    ├── Timestamp atual <= 60s da abertura da vela M15
    └── SE NÃO ──> VETO (SKIP: Timeout da Janela M15)

[7] PROPOSTA DA CORRETORA VIÁVEL?
    ├── Payout líquido cotado válido
    ├── EV > 0 com margem >= 0.01 acima do break-even
    └── SE NÃO ──> VETO (SKIP: Sem Edge Economico)
```

---

## 3. Protocolo de Gestão de Risco e Sizing

- **Tamanho de Lote Normal (EXPLORE):**
  $$	ext{Stake} = 0,01 	imes 	ext{Saldo Disponível}$$
- **Recuperação Linear (RECOVER):**
  $$	ext{Stake} = \min\left(rac{	ext{Prejuízo Pendente}}{	ext{Payout Cotado}}, 0,05 	imes 	ext{Saldo Disponível}ight)$$
- **Cap de Proteção:** Teto máximo de 5% por ordem.
- **Cooldown Pós-Loss:** Aguardo de 1 vela M15 (`loss_cooldown_candles=1`), estendendo para 2 velas em perdas consecutivas, liberando exatamente na abertura da próxima vela qualificada.
