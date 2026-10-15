# Configuração operacional SSOT

`config/settings.json` define as opções de runtime. Parsers em `app/src/domain/config_knobs.py` e nas camadas consumidoras validam os blocos necessários. Uma mudança de parâmetro operacional exige ajuste do parser, teste e documentação.

## Contrato ativo

| Bloco | Valor atual |
|---|---|
| `data_handler` | Micro/MINI M5 de 300 s; macro D1 de 86.400 s |
| `deep_learning` | TCN, lookback 32, 25.000 barras M5 no treino, 768 na inferencia, label `spot_forward`, horizonte 1 barra, `online_training=false` |
| `anchor`, `symbols`, `deep_learning.train_symbols` | Runtime ancorado em R_25 e descoberta de M5 aprovados; treino TCN explícito de R_10 e R_25. R_10 M15 aprovado não habilita compra M5. |
| `deep_learning.training_quality` | Um único bloco: diagnóstico linear (`min_linear_preflight_acc=0`), piso de seleção de checkpoint e detecção de colapso de classe; não qualifica deploy |
| `deep_learning.asymmetric_payout_loss` | `false`: BCE simétrica para preservar a probabilidade CALL; payout entra no EV da proposta |
| `deep_learning.checkpoint_max_stake_pct` | **0.01**, teto inicial de 1% da banca por ordem |
| `orchestrator.execution` | Ciclo M5, quatro vetos extremos, sem vetos redundantes de ruído/tendência/conformal/compressão, `force_trade_every_cycle=false`, `require_quote_edge=true`, `min_payout_rate=0` |
| `risk_management.kelly` | Kelly fracionário; payout de referência 0,85; stop win 4,31% |
| `risk_management.soft_recovery` | `cover_enabled=true`, `cover_multiple=1.0`, amortização 1/1, cap `cover_l0` 3,5% antes do teto do checkpoint |
| `infra.loss_classifier` | FLIP após auto aprendizado, `flip_min_n_train=1`, jovem `p_eff>=0.58`, maduro `p_eff>=0.70` |

O checkpoint passa por verificação técnica de pesos, normalização e geometria. Não existe qualificação estatística de deploy. Um checkpoint inválido impede a operação. Com checkpoint válido, o teto de 1% é aplicado também a `cover_l0`, na preparação do cluster e imediatamente antes da proposta. Limites menores ainda podem reduzir a stake.

`inference_history_bars=768` preserva na inferencia o contexto exigido pela diferenciacao fracionaria e pela escala causal das features aprendidas com historico longo. A sincronizacao inicial carrega esse piso mais warmup. Esse ajuste de janela nao muda o label nem o checkpoint e nao comprova maior acuracia por si so.

No `engine_mode=train`, a lista `deep_learning.train_symbols` substitui temporariamente `symbols` e a ancora para o stream de dados; a descoberta automatica de modelos aprovados fica fora desse processo. Na execucao, continuam valendo a ancora R_25 e a descoberta de modelos M5 aprovados. Isso evita treinar um simbolo sem ter sincronizado seu historico.

A política mantém stop win, **sem stop loss e sem teto acumulado de perda**. O bloqueio de compra por dados, checkpoint, cotação ou edge continua ativo. DEMO e REAL usam o mesmo contrato operacional.

Na proposta final, o motor calcula `quote_ev = p(lado) × (1 + payout_rate_cotado) − 1` com a taxa líquida do broker. `min_payout_rate=0` desativa o piso fixo que rejeitava cotações lucrativas abaixo de 0,80. O fluxo TCN exige `min_edge_execute=0` e `quote_safety_margin=0.01`. Com `indicator_consensus.multi_period_trading=true`, o sinal aprovado M5/M15/H1 da mesma versão, ativo, período e vela atual determina a direção. `broker_ev_promotion=true` exige em REAL 200 liquidações DEMO confirmatórias da mesma identidade e limite inferior unilateral de 95% do retorno médio acima de zero; a cotação REAL exige EV positivo e margem inclusive em recuperação. Compras exploratórias DEMO não contam. O piloto DEMO usa stake mínima, respeitando o teto de 1% da banca. A rota REAL do indicador usa `stake_pct=0.01`. Proposta sem preço, payout ou probabilidade válidos permanece bloqueada.

Se o broker rejeitar explicitamente o ID com `Unknown contract proposal`, somente o piloto DEMO pode tentar uma compra por parâmetros na mesma vela M5, uma vez e com preço máximo igual à stake. REAL não usa esse caminho porque a cotação da nova compra não pode ser revalidada antes da execução.

`skip_tcn_noise_discord`, `skip_trend_discord`, `conformal_uncertainty_gate` e `veto_compression_doji` estão desligados. O classificador de regime ainda produz telemetria; os quatro vetos de mercado extremos e a checagem da cotação final permanecem. Com `require_quote_edge=true`, ausência ou não finitude da probabilidade bloqueia a compra antes da proposta. A redução de skips não transforma probabilidade próxima de 0,5 em EV positivo: sem vantagem estimada à cotação real, a ordem continua bloqueada.

Consulte [treino TCN](engineering-deep-learning.md), [arquitetura](arquitetura.md) e [gates de indicadores](engineering-indicator-gates.md).
