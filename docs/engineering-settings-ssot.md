# Configuracao operacional SSOT

`config/settings.json` define a rota de compras. O bloco `pullback_strategy` deve ter `enabled=true`, `variant=senior_institutional`, `period_seconds=900`, `duration=15`, `entry_window_seconds=60` e a versao produzida por `version_for_variant`. A ancora configurada e `OTC_SSMI`. O stream micro/MINI do motor ainda usa 300 s para pesquisa legada; ele nao fornece direcao operacional.

`never_skip`, `allow_closed_markets`, `organic_direction`, `stream_fallback` e `invert_signal` ficam desativados (`max_fallback_candidates=0`). Com `operate_all_signals=true` e `dynamic_rotation=true`, o motor opera todos os sinais M15 institucionais validados pelo indicador nos 12 indices mundiais OTC, rotacionando dinamicamente quando a ancora nao apresentar sinal valido. A rota pullback exige a ultima vela M15 fechada, a janela de entrada, a versao exata do setup e o lado publicado. H1 e D1 nao geram contratos.

O indicador usa apenas OHLC publico da Deriv para a serie do setup. A referencia da sessao UTC usa preco tipico por vela, sem volume; nao e VWAP. As killzones BRT sao 03h-08h, 10h-13h e 14h-16h. O perfil horario do treino e diagnostico e nao substitui essas janelas.

`make app-train` registra o proxy de velas e seu ranking, sem escrever em `config/settings.json` ou promover `pullback_registry`. O servico do indicador cria novas versoes como `piloto` e a promocao para `ativo` depende de 200 compras e liquidacoes DEMO unicas, atribuidas a ativo, periodo e versao, com limite inferior unilateral de 95% positivo. A rota REAL verifica essa evidencia novamente e exige EV positivo com margem de 0,01 na proposta final.

A stake base e 1% da banca disponivel. A recuperacao usa o payout liquido cotado com amortizacao suave em etapas e teto de 5% por ordem (`loss_cooldown_candles=1`, `consecutive_loss_cooldown_candles=2` alinhados a fronteira da vela M15). Nao ha stop loss nem teto acumulado de perda. TCN, meta e loss-classifier nao alteram o lado nem a stake desta rota.

Para alterar um parametro operacional, atualize parser/consumidor, testes, este documento, regras e skills. Consulte [setup](strategy-pullback.md) e [arquitetura](arquitetura.md).
