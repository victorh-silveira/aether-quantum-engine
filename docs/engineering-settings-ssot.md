# Configuração operacional SSOT

`config/settings.json` define as opções de runtime. Parsers em `app/src/domain/config_knobs.py` e nas camadas consumidoras validam os blocos necessários. Uma mudança de parâmetro operacional exige ajuste do parser, teste e documentação.

## Contrato ativo

| Bloco | Valor atual |
|---|---|
| `data_handler` | Micro/MINI M5 de 300 s; macro D1 de 86.400 s |
| `deep_learning` | TCN, lookback 32, 25.000 barras M5, label `spot_forward`, horizonte 1 barra, `online_training=false` |
| `deep_learning.training_quality` | Métricas de treino e detecção de colapso de classe; não qualificam deploy |
| `deep_learning.checkpoint_max_stake_pct` | **0.01**, teto inicial de 1% da banca por ordem |
| `orchestrator.execution` | Ciclo M5, `four_market_vetoes=true`, `market_direction_trigger=true`, `force_trade_every_cycle=false`, `require_quote_edge=true`, `min_payout_rate=0` |
| `risk_management.kelly` | Kelly fracionário; payout de referência 0,85; stop win 4,31% |
| `risk_management.soft_recovery` | `cover_enabled=true`, `cover_multiple=1.0`, amortização 1/1, cap `cover_l0` 3,5% antes do teto do checkpoint |
| `infra.loss_classifier` | FLIP após auto aprendizado, `flip_min_n_train=1`, jovem `p_eff>=0.58`, maduro `p_eff>=0.70` |

O checkpoint passa por verificação técnica de pesos, normalização e geometria. Não existe qualificação estatística de deploy nem simulação de liquidação por closes M5. Um checkpoint inválido impede a operação. Com checkpoint válido, o teto de 1% é aplicado também a `cover_l0`, na preparação do cluster e imediatamente antes da proposta. Limites menores ainda podem reduzir a stake.

A política mantém stop win, **sem stop loss e sem teto acumulado de perda**. O bloqueio de compra por dados, checkpoint, cotação ou edge continua ativo. DEMO e REAL usam o mesmo contrato operacional.

Na proposta final, o motor calcula `quote_ev = p(lado) × (1 + payout_rate_cotado) − 1` com a taxa líquida do broker. `min_payout_rate=0` desativa o piso fixo que rejeitava cotações lucrativas abaixo de 0,80; `min_edge_execute=0` e `quote_safety_margin=0.01` continuam exigidos antes da compra. Proposta sem preço ou payout válido permanece bloqueada.

Consulte [treino TCN](engineering-deep-learning.md), [arquitetura](arquitetura.md) e [gates de indicadores](engineering-indicator-gates.md).
