# Medallion — indices sinteticos Deriv (M1)

Motor na branch `feat/synthetic-indices-m5`: volatilidade sintetica Deriv com ancora **R_100** (Volatility 100 Index), clusters de baixa volatilidade (US) e alta volatilidade (EU), contratos **RISE/FALL 1 minuto** e decisao **Google Gemini** com guardrails quantitativos.

## Universo

| Bloco | Simbolos | Papel |
|-------|----------|--------|
| Ancora | `R_100` | Marcapasso de regime entre blocos VOL |
| US (baixa vol) | `R_10`, `R_25`, `R_50` | Cluster executavel em `risk_on` / divergencia US |
| EU (alta vol) | `R_75`, `1HZ50V`, `1HZ100V` | Cluster executavel em `risk_off` / divergencia EU |

## Camadas

| Camada | Funcao |
|--------|--------|
| Macro M1 | Tags `risk_on`, `risk_off`, `divergence_*`, `indefinido` a partir dos clusters |
| StatArb | Z-Score por indice no cluster ativo; selecao do melhor simbolo |
| LLM | `US_CLUSTER` e `EU_CLUSTER` independentes; JSON com probabilidade |
| Execucao | `cluster_invert_llm_side`: inverte CALL/PUT da LLM no indice executado |
| Pos-loss | `repeat_loss_setup` com fallback para outro indice do cluster; cooldown configuravel |

## Configuracao

Arquivo: `config/settings.json` nesta branch. Parametros-chave:

- `anchor` / `strategy.correlation.anchor`: `R_100`
- `risk_management.params.duration`: `1` (`duration_unit`: `m`)
- `orchestrator.cycle_interval_seconds`: `45`
- `strategy.correlation.cluster_invert_llm_side`: `true`
- `orchestrator.cluster_repeat_loss_block_cycles`: janela de veto apos loss no mesmo setup

Documentacao de arquitetura: [`arquitetura-synthetic.md`](arquitetura-synthetic.md).
