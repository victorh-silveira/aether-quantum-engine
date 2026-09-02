## [2.50.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.50.0...v2.50.1) (2026-09-02)

### Correcoes de Bug

* **domain:** mapeia volatilidade alvo de 75 pct para 1HZ75V ([d54a705](https://github.com/victorh-silveira/aether-quantum-engine/commit/d54a705edd608e6be91045f711bb95d875a78b64))

### Refatoracoes Tecnicas

* **config:** migra operacao e timeframes de M15 para M5 ([6efcbcb](https://github.com/victorh-silveira/aether-quantum-engine/commit/6efcbcb7cb013d886c26e28cdf26f73aa1216a47))

### Documentacao

* **all:** moderniza rules, skills, docs e arquitetura para padrao senior ([a9872e3](https://github.com/victorh-silveira/aether-quantum-engine/commit/a9872e3e5e91fc29c82bfb3f3750d2e6858b9e52))

## [2.50.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.49.0...v2.50.0) (2026-09-02)

### Funcionalidades

* **orchestrator:** aplicar bloqueio estrito de operacao na zona cinzenta ([963302d](https://github.com/victorh-silveira/aether-quantum-engine/commit/963302d5c2ba669be1b44c886ef1a55ac3453c26))

### Correcoes de Bug

* **infra:** adicionar capacidades de troca de usuario para redis alpine ([64db86e](https://github.com/victorh-silveira/aether-quantum-engine/commit/64db86ea3b9551eded216f51ed00803e5474f622))

## [2.49.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.48.0...v2.49.0) (2026-09-02)

### Funcionalidades

* **infra:** implementar blindagem avancada e gestao de processos docker ([4a58107](https://github.com/victorh-silveira/aether-quantum-engine/commit/4a58107391741b8c996a172e92285561d66f937a))

### Melhorias de Performance

* **infra:** acelerar e calibrar treinamento do loss-classifier e meta-regressor ([6f084ca](https://github.com/victorh-silveira/aether-quantum-engine/commit/6f084caab4ce224424effab17e6517ae64207f17))
* **infra:** otimizar infraestrutura de dados e microservicos ml ([c37ab42](https://github.com/victorh-silveira/aether-quantum-engine/commit/c37ab42620d94627cf099d0d5398b3392151520f))

## [2.48.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.47.0...v2.48.0) (2026-09-02)

### Funcionalidades

* **infra:** otimizar piso de retreino do loss-classifier para 4 amostras ([632497c](https://github.com/victorh-silveira/aether-quantum-engine/commit/632497c8c068fe72b93f680960aecf5e04904b95))

## [2.47.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.46.0...v2.47.0) (2026-09-01)

### Funcionalidades

* **engine:** implementar rotulagem quantum_multi_barrier e barreiras assimetricas ([3ffe09a](https://github.com/victorh-silveira/aether-quantum-engine/commit/3ffe09a3fd50485fc0f386fe2fd1c21db2344586))

## [2.46.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.45.4...v2.46.0) (2026-09-01)

### Funcionalidades

* **engine:** otimizar predicao de candlestick e calibrar exhaustion filter ([41e37ab](https://github.com/victorh-silveira/aether-quantum-engine/commit/41e37ab8b5d9ecc2a45515a80cc072fb0d678f8b))

### Documentacao

* **all:** sincronizar rules e AGENTS com novas diretrizes ([7e3b237](https://github.com/victorh-silveira/aether-quantum-engine/commit/7e3b237116aebf20b392b8e02791a889bb1965f4))

## [2.45.4](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.45.3...v2.45.4) (2026-09-01)

### Correcoes de Bug

* **risk:** blindar flip anti-loss e calibrar amortizacao de soft recovery ([96c58ca](https://github.com/victorh-silveira/aether-quantum-engine/commit/96c58ca33f895c04c3c15c2ab07ac0f5150914cd))

## [2.45.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.45.2...v2.45.3) (2026-08-31)

### Correcoes de Bug

* **risk:** calibrar piso de winrate no stop win inicial ([d46eee3](https://github.com/victorh-silveira/aether-quantum-engine/commit/d46eee31b200cb04adc9af076956a11700e070b5))

## [2.45.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.45.1...v2.45.2) (2026-08-31)

### Melhorias de Performance

* **risk:** otimizar damping de proximidade e alinhamento de stop win ([ec74876](https://github.com/victorh-silveira/aether-quantum-engine/commit/ec7487651491fc110ef2371d199c3e605b85502b))

## [2.45.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.45.0...v2.45.1) (2026-08-31)

### Correcoes de Bug

* **orchestrator:** encerrar motor automaticamente ao disparar stop win ([e19bf99](https://github.com/victorh-silveira/aether-quantum-engine/commit/e19bf9928b0ef8913a7b6de541e8589f3ae95efc))

## [2.45.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.44.0...v2.45.0) (2026-08-31)

### Funcionalidades

* **risk:** adicionar tolerancia de 95% para gatilho de stop win ([652f425](https://github.com/victorh-silveira/aether-quantum-engine/commit/652f425f0c08bd03598fab177703c13e4a1c33a2))

## [2.44.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.43.1...v2.44.0) (2026-08-31)

### Funcionalidades

* **config:** calibrar limiares de confianca para selecao de alta conviccao ([3797f54](https://github.com/victorh-silveira/aether-quantum-engine/commit/3797f54e16a8cb5f6da8475d9f7af44aec9cc4ac))

## [2.43.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.43.0...v2.43.1) (2026-08-31)

### Correcoes de Bug

* **scripts:** calibrar volume de barras do treino meta para 365 barras ([8f4375b](https://github.com/victorh-silveira/aether-quantum-engine/commit/8f4375b902189a79727f32c925f1656bc02f6333))

## [2.43.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.42.0...v2.43.0) (2026-08-31)

### Funcionalidades

* **orchestrator:** priorizar inversao para candle em discordancias de EMA ([ceb78a7](https://github.com/victorh-silveira/aether-quantum-engine/commit/ceb78a72b12d0ee0c24d00ef1a9ae0d480321266))

## [2.42.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.41.0...v2.42.0) (2026-08-30)

### Funcionalidades

* **orchestrator:** habilitar inversao inteligente de direcao para o candle ([dcf2c13](https://github.com/victorh-silveira/aether-quantum-engine/commit/dcf2c13fcadb1593fe4b3167d61ddec9b8d650b6))

## [2.41.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.40.0...v2.41.0) (2026-08-30)

### Funcionalidades

* **risk:** ativar anti_loss_hard_skip para vetar trades contra tendencia com P(loss) alto ([810fe76](https://github.com/victorh-silveira/aether-quantum-engine/commit/810fe7650aad20354f6c469b6047eeaa4eb68d58))

## [2.40.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.39.0...v2.40.0) (2026-08-30)

### Funcionalidades

* **risk:** permitir execucao de sinal com edge negativo via soft Kelly ([fe7c7cd](https://github.com/victorh-silveira/aether-quantum-engine/commit/fe7c7cdbff4999a5104ac7782bc48859a54bfc57))

## [2.39.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.38.0...v2.39.0) (2026-08-30)

### Funcionalidades

* **risk:** permitir execucao de sinal puro TCN com soft Kelly em EV fraco ([9ae5d21](https://github.com/victorh-silveira/aether-quantum-engine/commit/9ae5d21cd31ec1eb544d9d98265dd330fb0e4b9c))

## [2.38.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.37.0...v2.38.0) (2026-08-30)

### Funcionalidades

* **engine:** garantir soberania estrita de 100% da direcao TCN sem inversoes de fusao ([fea33c9](https://github.com/victorh-silveira/aether-quantum-engine/commit/fea33c943528a9f04ee982a00f7824c3aa221200))

### Refatoracoes Tecnicas

* **repo:** renomear rules e skills de mercado para v75 ([f7a31f1](https://github.com/victorh-silveira/aether-quantum-engine/commit/f7a31f15dd975843be580ca65719cfe590229148))

## [2.37.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.36.0...v2.37.0) (2026-08-30)

### Funcionalidades

* **config:** expandir historico de treino para 365 velas D1 e 500 M15 ([aabeffe](https://github.com/victorh-silveira/aether-quantum-engine/commit/aabeffee30c655b7348dd9e6953a823ae173cef6))

## [2.36.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.35.0...v2.36.0) (2026-08-30)

### Funcionalidades

* **engine:** implementar Triple Barrier Method para previsao de alta precisao ([5e2a126](https://github.com/victorh-silveira/aether-quantum-engine/commit/5e2a1261965f1b2c49ae35d9c478c6cdeea7ee95))

## [2.35.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.34.0...v2.35.0) (2026-08-30)

### Funcionalidades

* **domain:** migrar ativo operacional para Volatility 75 (1s) ([02ef879](https://github.com/victorh-silveira/aether-quantum-engine/commit/02ef8792860c506e85eb1bfe1e62ad88bbbcec62))

## [2.34.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.33.0...v2.34.0) (2026-08-27)

### Funcionalidades

* **orchestrator:** desativar gates e skips de anti-loss e discordancia ([3c26e33](https://github.com/victorh-silveira/aether-quantum-engine/commit/3c26e33f9cd937d5ee97f3f62dba84edf2934b71))

## [2.33.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.32.0...v2.33.0) (2026-08-27)

### Funcionalidades

* **orchestrator:** atualizar meta stop win 4.31, cadencia e anti-loss ([809606e](https://github.com/victorh-silveira/aether-quantum-engine/commit/809606ea424ddbe52adf22fcab1496a89d332f30))

## [2.32.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.31.5...v2.32.0) (2026-08-26)

### Funcionalidades

* **orchestrator:** alinhar telemetria candle para granularidade micro m15 e acelerar loss clf bootstrap ([217c977](https://github.com/victorh-silveira/aether-quantum-engine/commit/217c9779c252014f22160d1aba2b026eed8f15d8))
* **orchestrator:** inverter direcao inteligentemente para a vela em discordancia ao inves de vetar ([351bc5a](https://github.com/victorh-silveira/aether-quantum-engine/commit/351bc5ac73ae598f8c205900d733e02516f93b6b))

### Correcoes de Bug

* **orchestrator:** sincronizar ssot de loss classifier e inversao inteligente anti-loss ([488b846](https://github.com/victorh-silveira/aether-quantum-engine/commit/488b846f85a8965652473b3f132f459f84af2525))

## [2.31.5](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.31.4...v2.31.5) (2026-08-26)

### Correcoes de Bug

* **orchestrator:** calibrar tolerancia dinamica de microestrutura ema para escala sp500 ([87f5dcf](https://github.com/victorh-silveira/aether-quantum-engine/commit/87f5dcfdbe57de2dda976d41d485f590b557cfd1))

## [2.31.4](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.31.3...v2.31.4) (2026-08-26)

### Correcoes de Bug

* **scripts:** calibrar proporcao de amostras do dataset meta para M15 ([008f9db](https://github.com/victorh-silveira/aether-quantum-engine/commit/008f9dbd5db9ab743058fbd180399e9a480f7fb2))

## [2.31.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.31.2...v2.31.3) (2026-08-26)

### Correcoes de Bug

* **scripts:** calibrar piso de barras e testes do meta para M15 ([75211c0](https://github.com/victorh-silveira/aether-quantum-engine/commit/75211c000ddd58db4279843b58640b548bc2c43d))

### Documentacao

* **all:** atualizar documentacao e readmes para OTC_SPC e M15 ([b899ba7](https://github.com/victorh-silveira/aether-quantum-engine/commit/b899ba75037c04df08c41c9d81c0a94d4f72c3f3))

## [2.31.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.31.1...v2.31.2) (2026-08-25)

### Melhorias de Performance

* **config:** otimizar lookback e historico de treino para OTC_SPC ([61c121d](https://github.com/victorh-silveira/aether-quantum-engine/commit/61c121d815cf11af3251b0fafdd2f6824aad0add))

## [2.31.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.31.0...v2.31.1) (2026-08-25)

### Correcoes de Bug

* **domain:** atualizar simbolo oficial SP 500 para OTC_SPC ([d330408](https://github.com/victorh-silveira/aether-quantum-engine/commit/d330408dfb98d9da5304baa5717e939f3a3b083a))

## [2.31.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.30.0...v2.31.0) (2026-08-25)

### Funcionalidades

* **all:** migracao completa para SP 500 M15 e Kelly Single-Strike 1% ([3873e72](https://github.com/victorh-silveira/aether-quantum-engine/commit/3873e7217ac0f732e528dbb36818cbedfc10904f))

## [2.30.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.29.0...v2.30.0) (2026-08-25)

### Funcionalidades

* **orchestrator:** desativar trava de vela na fusao e tolerar pullbacks na EMA ([32abc43](https://github.com/victorh-silveira/aether-quantum-engine/commit/32abc43d9159c67c57bfb0a63cb1c10c6689d34b))
* **orchestrator:** harmonizar limiares operacionais e integrar edge pos-fusao ([b988d11](https://github.com/victorh-silveira/aether-quantum-engine/commit/b988d1109d96701d5d1348b52b28fc78be294965))
* **orchestrator:** implementar filtros de microestrutura M5, regras estritas de gates e sincronizar skills de trader senior ([40a4e6b](https://github.com/victorh-silveira/aether-quantum-engine/commit/40a4e6b75af4d0911aca039d7efe442f4ab1a75e))

### Correcoes de Bug

* **config:** elevar piso minimo de borda para seis centesimos ([42dffc4](https://github.com/victorh-silveira/aether-quantum-engine/commit/42dffc4df9b07c876bf5fe5e6984e651605159d5))
* **dl:** ajustar suavizacao de rotulo temperatura e paciencia ([4afb15d](https://github.com/victorh-silveira/aether-quantum-engine/commit/4afb15d4424316b6bf7082a232b0bfe077d71414))
* **orchestrator:** bloquear execucao em borda negativa e fusao neutra ([cd7b491](https://github.com/victorh-silveira/aether-quantum-engine/commit/cd7b49143e672636d3d8f79c68b033565b0774b7))
* **orchestrator:** flexibilizar limiar de borda e acelerar deteccao de tendencia ([07b8338](https://github.com/victorh-silveira/aether-quantum-engine/commit/07b833886a9d2e7f65f0a6aa00051ee567a72733))
* **repo:** alinhar QA local ao CI e SSOT de producao ([29e10ba](https://github.com/victorh-silveira/aether-quantum-engine/commit/29e10ba11176a01b9933ee59a56196d920524503))

### Refatoracoes Tecnicas

* **orchestrator:** modularizar anti-loss em helper dedicado com 100% cov ([dae9791](https://github.com/victorh-silveira/aether-quantum-engine/commit/dae979135706b6d21ab44173558053fb09c46738))

## [2.29.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.28.0...v2.29.0) (2026-08-21)

### Funcionalidades

* **infra:** elevar ready_n do loss_classifier para 30 amostras ([5363d3c](https://github.com/victorh-silveira/aether-quantum-engine/commit/5363d3ca9c0353b1488d89a6956aa64d1cdd40d8))

## [2.28.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.27.1...v2.28.0) (2026-08-21)

### Funcionalidades

* **config:** restaurar amortecimento de Brier com label smoothing e focal loss ([b69c1ec](https://github.com/victorh-silveira/aether-quantum-engine/commit/b69c1ec78cde0c8106c0f9d739944ea272e39803))

## [2.27.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.27.0...v2.27.1) (2026-08-21)

### Correcoes de Bug

* **engine:** alinhar calculo de gray no log do teacher ao SSOT ([f4e4651](https://github.com/victorh-silveira/aether-quantum-engine/commit/f4e4651eeb62a6ba6f058fb9b9f83692c2b434d0))

## [2.27.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.26.0...v2.27.0) (2026-08-21)

### Funcionalidades

* **config:** calibrar thresholds empiricos 0.48/0.32 ([7c5e2e0](https://github.com/victorh-silveira/aether-quantum-engine/commit/7c5e2e0356f36de4cb3d752db92387b7c195888b))

## [2.26.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.25.0...v2.26.0) (2026-08-21)

### Funcionalidades

* **config:** calibrar thresholds 0.55/0.45 e label_smoothing 0.015 ([43ffb5e](https://github.com/victorh-silveira/aether-quantum-engine/commit/43ffb5ea5b75ff16ffb10fd6e7a477573124170b))

## [2.25.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.24.0...v2.25.0) (2026-08-21)

### Funcionalidades

* **config:** reequilibrar thresholds de confianca e regularizar Optuna ([70a4215](https://github.com/victorh-silveira/aether-quantum-engine/commit/70a42152f0cce8f4b3a5689849b09cdb92d5e217))

## [2.24.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.23.1...v2.24.0) (2026-08-20)

### Funcionalidades

* **engine:** calibrar label smoothing para 0.04 e expandir validacao meta para 25% ([4b47b4c](https://github.com/victorh-silveira/aether-quantum-engine/commit/4b47b4c1bd91cf42891cc27aff8511f76c3acd0c))

## [2.23.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.23.0...v2.23.1) (2026-08-20)

### Correcoes de Bug

* **config:** ajustar label_smoothing para 0.08 visando Brier estavel com stride=5 ([cb358ea](https://github.com/victorh-silveira/aether-quantum-engine/commit/cb358ea8006b243184f76651a9f45e13f40cf803))

## [2.23.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.22.2...v2.23.0) (2026-08-20)

### Funcionalidades

* **engine:** adicionar calibrador puro de temperatura nos candidatos de ajuste ([1c99953](https://github.com/victorh-silveira/aether-quantum-engine/commit/1c999533235c0f0a847df0983765211bb31db17f))

## [2.22.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.22.1...v2.22.2) (2026-08-20)

### Correcoes de Bug

* **scripts:** usar arrays contiguos e thread unica no LightGBM para estabilidade no Windows ([a20c630](https://github.com/victorh-silveira/aether-quantum-engine/commit/a20c63064d336ca08ef22e914f4e6884b25b42c9))

## [2.22.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.22.0...v2.22.1) (2026-08-20)

### Correcoes de Bug

* **scripts:** usar gather_every no Polars para subamostragem stride=5 ([2369a22](https://github.com/victorh-silveira/aether-quantum-engine/commit/2369a2209b8023ad9490b5160d4c39682d7dfec9))

## [2.22.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.21.1...v2.22.0) (2026-08-20)

### Funcionalidades

* **engine:** implementar subamostragem stride=H na validacao para estrita independencia IID ([83d75cd](https://github.com/victorh-silveira/aether-quantum-engine/commit/83d75cd4df7f243277d1fb9b5eaa2f1cb23b9d0c))

## [2.21.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.21.0...v2.21.1) (2026-08-20)

### Correcoes de Bug

* **config:** aplicar label smoothing e ajustar temperature scaling para controle de Brier ([d86a327](https://github.com/victorh-silveira/aether-quantum-engine/commit/d86a327446686020293a79b65f42d22fe18ec08b))

## [2.21.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.20.0...v2.21.0) (2026-08-20)

### Funcionalidades

* **infra:** expandir dataset para 5000 barras e regularizar Optuna ([26f324f](https://github.com/victorh-silveira/aether-quantum-engine/commit/26f324f49427a2b9506a14552dee645dc96f68d5))

## [2.20.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.19.0...v2.20.0) (2026-08-20)

### Funcionalidades

* **engine:** expandir buffer de treino para 5000 barras e calibrar anti-overfit ([9736ec2](https://github.com/victorh-silveira/aether-quantum-engine/commit/9736ec24b7a978c274af16a66b5e374bdf2a2ee5))

## [2.19.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.18.3...v2.19.0) (2026-08-20)

### Funcionalidades

* **config:** padronizar campo amostral em multiplos de 10 e N>=1000 ([d6f6a15](https://github.com/victorh-silveira/aether-quantum-engine/commit/d6f6a154f172ee65b5252fd333c7d209bfe303ec))

### Correcoes de Bug

* **infra:** atualizar artefato do meta modelo lgbm otimizado ([115fb3c](https://github.com/victorh-silveira/aether-quantum-engine/commit/115fb3ce6e55ca114f4cfac42c47d8fb45058bda))

## [2.18.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.18.2...v2.18.3) (2026-08-20)

### Correcoes de Bug

* **engine:** preservar estado sharp de validacao e calibrar patience ([860aa29](https://github.com/victorh-silveira/aether-quantum-engine/commit/860aa2937a38eab2259d705fec2d2c0863131673))

## [2.18.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.18.1...v2.18.2) (2026-08-20)

### Correcoes de Bug

* **infra:** formatar amount com duas casas decimais no bulk purchase ([962c11f](https://github.com/victorh-silveira/aether-quantum-engine/commit/962c11f5783549a6943f408da9129d903d409fed))

## [2.18.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.18.0...v2.18.1) (2026-08-19)

### Correcoes de Bug

* **orchestrator:** reduzir peso micro bar na fusao EV ([1d13e42](https://github.com/victorh-silveira/aether-quantum-engine/commit/1d13e42df19a9badfd896a2a45a68c2993a77139))

## [2.18.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.17.0...v2.18.0) (2026-08-19)

### Funcionalidades

* **orchestrator:** ativar protecao anti-loss contra discordancia de vela ([5842b39](https://github.com/victorh-silveira/aether-quantum-engine/commit/5842b3985cabc8c51356f8e70b2d16248b5acaa3))

## [2.17.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.16.2...v2.17.0) (2026-08-19)

### Funcionalidades

* **orchestrator:** desbloquear execucao com kelly real na fusao ([0997232](https://github.com/victorh-silveira/aether-quantum-engine/commit/09972324763334f97c62122270f2ff51a920b892))

## [2.16.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.16.1...v2.16.2) (2026-08-19)

### Correcoes de Bug

* **app:** sincroniza resolve_label_mode com supertrend_atr ([360193e](https://github.com/victorh-silveira/aether-quantum-engine/commit/360193e6d046a38b991636119124e028c2edd670))

## [2.16.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.16.0...v2.16.1) (2026-08-19)

### Correcoes de Bug

* **scripts:** corrige recursao infinita no fallback do launch-train ([d4fcd3b](https://github.com/victorh-silveira/aether-quantum-engine/commit/d4fcd3b169bb73b9d54d3e49224f03378bb9d01e))

## [2.16.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.15.0...v2.16.0) (2026-08-19)

### Funcionalidades

* **app:** ativa aceleracao cuda rtx 4060 e recalibra early stopping ([d03bd80](https://github.com/victorh-silveira/aether-quantum-engine/commit/d03bd80f3d54da888be268679ca94a1c62265e42))
* **infra:** aprimora telemetria de hidratacao m1 e h2 no timescale ([d3a536e](https://github.com/victorh-silveira/aether-quantum-engine/commit/d3a536e21c9213ea98516b6bd5e14337bd193f98))

## [2.15.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.14.0...v2.15.0) (2026-08-19)

### Funcionalidades

* **config:** recalibra janelas de indicadores para contrato 5m ([29c5ecf](https://github.com/victorh-silveira/aether-quantum-engine/commit/29c5ecf7fca7c3a382a651e2191a0d251549a403))

## [2.14.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.13.0...v2.14.0) (2026-08-19)

### Funcionalidades

* **infra:** refatora modelos e fit adaptativo dos microservicos ml ([3335dd5](https://github.com/victorh-silveira/aether-quantum-engine/commit/3335dd5f24e1935d25479cb2224670fa4867e5fb))

## [2.13.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.12.2...v2.13.0) (2026-08-19)

### Funcionalidades

* **app:** migra para rotulagem supertrend_atr e treino focado 5m ([62b3c59](https://github.com/victorh-silveira/aether-quantum-engine/commit/62b3c59a06671a3ab2734ab653d365c9519d38a1))

## [2.12.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.12.1...v2.12.2) (2026-08-19)

### Correcoes de Bug

* **engine:** exige corpo minimo da vela no anti-loss ([f8985b5](https://github.com/victorh-silveira/aether-quantum-engine/commit/f8985b5419498e40dfc6b195b9935b2f9f6f4d6e))
* **orchestrator:** estabiliza treino, gates de execucao e makefile ([1695670](https://github.com/victorh-silveira/aether-quantum-engine/commit/1695670d2e424245b4d548cbfe559831db2271a9))
* **release:** fixa versao do conventionalcommits para compatibilidade com sem-rel ([a5264e2](https://github.com/victorh-silveira/aether-quantum-engine/commit/a5264e2692be49b1548a6eedc9cd2caa77e6f08d))

## [2.12.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.12.0...v2.12.1) (2026-08-17)

## [2.12.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.11.0...v2.12.0) (2026-08-17)

## [2.11.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.10.0...v2.11.0) (2026-08-17)

## [2.10.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.9.0...v2.10.0) (2026-08-16)

## [2.9.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.6...v2.9.0) (2026-08-16)

## [2.8.6](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.5...v2.8.6) (2026-08-14)

## [2.8.5](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.4...v2.8.5) (2026-08-14)

### Removed
* **infra:** servidor de inferencia gRPC removido da stack Docker; TCN = eager/CUDA local; profiles `core`/`ml` apenas

## [2.8.4](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.3...v2.8.4) (2026-08-13)

## [2.8.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.2...v2.8.3) (2026-08-13)

## [2.8.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.1...v2.8.2) (2026-08-11)

## [2.8.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.8.0...v2.8.1) (2026-08-11)

## [2.8.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.7.0...v2.8.0) (2026-08-10)

## [2.7.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.6.1...v2.7.0) (2026-08-10)

## [2.6.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.6.0...v2.6.1) (2026-08-10)

## [2.6.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.5.0...v2.6.0) (2026-08-10)

## [2.5.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.4.0...v2.5.0) (2026-08-10)

## [2.4.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.3.0...v2.4.0) (2026-08-10)

## [2.3.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.2.0...v2.3.0) (2026-08-10)

## [2.2.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.1.0...v2.2.0) (2026-08-10)

## [2.1.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v2.0.0...v2.1.0) (2026-08-09)

## [2.0.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.82.0...v2.0.0) (2026-08-08)
* **loss_clf_calibrate:** floor FLIP **0.90**; fit `class_weight=balanced` + `min_child_samples=15`; retrain pos-LOSS bloqueado se buffer LOSS-heavy (`loss/n>0.60` ou `win<8`) exceto saida bootstrap (≥1 WIN+≥1 LOSS); vetor 24D clip edge idx10 + `micro_tick_acceleration` idx19; Prob/Cal e `p_loss` com **5** casas; limpeza+bootstrap embutidos em `make docker-rebuild` / `docker-reset` (`COLD_START`, `p_loss=0.50`, veto off ate retrain real)
* **scale_mili_tape_chop:** `adapt_mili_tape_skip_chop` **true** — nao inverter TCN so com mili+tape em micro=chop (majority com lead permanece)
* **otc_spc/m15:** contrato/OHLC M15; ciclo/assinatura **60 s** (entrada a cada 1 m); payout live **0.72**; settle tolerancia **300 s**; watchdog stale **600 s**; SIDE_EQ/sample_size densos; price_zone off
* **loss_clf_flip:** `hard_p_loss_floor` **0.90** → **FLIP** CALL↔PUT **ancorado no TCN** so com `veto_ready`; bootstrap devolve `p_loss=0.50` (sem FLIP); soft Kelly em `[0.65, 0.90)` com `veto_ready`; saida bootstrap no `/learn` com ≥1 WIN+≥1 LOSS
* **loss_clf_hard:** mandato **2026-08-07** — substituido por **loss_clf_flip** (floor **0.90**)
* **loss_clf_flip_legacy:** floor **0.80** + FLIP no bootstrap removidos apos anti-TCN em cold-start
* **regime_chop:** mandato **2026-08-07** — pausa M15 se ADX &lt; **0.22** e Hurst ∈ [**0.47**, **0.53**] (`SKIP:REGIME_CHOP`)
* **scale_majority:** `adapt_on_majority_votes` conta TCN/tape/mili/mini_pair/RSI; lado com mais votos adapta EXEC
* **stake_2pct:** explore piso **2%** banca (`neutral_bankroll_pct`/`min_stake_pct`); loss_clf soft nao esmaga U; RECOVER `cover_multiple` **2.0** (loss+win); tetos linear2/3 **5%**
* **remove_calib_gray:** remove `hold_calib_gray` / `hold_cal_margin` / `adapt_*_cal_margin` / `calib_gray_*` soft Kelly+teto / log `CALIB_GRAY`
* **risk:** piso explore **1%** / teto **5%**; loss-clf soft_max **2%** / soft_mult_high **0.40**
* **startup:** remove `startup_fetch_bars: 512` (era &lt; lookback 720 → SKIP data); piso = `min_dl_inference_len`; sync lean tambem em inferencia
* **infer:** piso `inference_history_bars` sem double-count de lookback (~864 vs ~1456); evita SKIP:DATA quando API R_10 entrega ~1.2k M15
* **risk:** pos-LOSS sem flip DIR_LOCK (lado fica no TCN); cover RECOVER sem damping de stop-win; loss-clf soft waivado com pending
