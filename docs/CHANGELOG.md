## 1.0.0 (2026-06-19)

### ⚠ BREAKING CHANGES

* migração para 100% deep learning PyTorch e recuperação Martingale cross-symbol em mercado único

### Funcionalidades

* **all:** backtest M15 assertivo, risco diario e guardrails macro ([91a815b](https://github.com/victorh-silveira/aether-quantum-engine/commit/91a815b3e13f60f12acd2f60e75778ec267bec62))
* **api:** migra autenticacao Deriv para PAT e App ID ([6688a74](https://github.com/victorh-silveira/aether-quantum-engine/commit/6688a74e5de63e3c94b783bfb5eb852172655349))
* **app:** adiciona confirmacao de tendencia SMA-20 no flip de recuperacao ([08c46a7](https://github.com/victorh-silveira/aether-quantum-engine/commit/08c46a7dc1cc3ad72a85472339e311942e1420ac))
* **app:** adiciona fallback de tendencia SMA-20 em execucao ([496725e](https://github.com/victorh-silveira/aether-quantum-engine/commit/496725e78792fa11dde24b01f2b45c557f30ba62))
* **app:** adiciona filtro de tendencia curto configuravel ([147cbbf](https://github.com/victorh-silveira/aether-quantum-engine/commit/147cbbfe309bacd5c12747332d3b5ebcd2ce926e))
* **app:** adiciona inclinacao no calculo de tendencia ([7a765f8](https://github.com/victorh-silveira/aether-quantum-engine/commit/7a765f8fd9bda5ec7607b5bc58b34fb77bfe971c))
* **app:** ajustar direcao em recovery e adicionar filtro rsi ([54b78cf](https://github.com/victorh-silveira/aether-quantum-engine/commit/54b78cf69b57458a798d6320066d6efe0f5498e9))
* **app:** remove bloqueios de cooldown e session pause para eliminar holds ([b84ad6c](https://github.com/victorh-silveira/aether-quantum-engine/commit/b84ad6c9ffdfcfd6024c6d91f98479e7f8a419a8))
* **app:** resetar saldo demo automaticamente se zerado ([14956a0](https://github.com/victorh-silveira/aether-quantum-engine/commit/14956a06f253786e92a0181121a1949e3e651d36))
* **config:** ajusta fracao de Kelly e fracao alvo de Martingale ([442d30c](https://github.com/victorh-silveira/aether-quantum-engine/commit/442d30c441e74ba40a899a99e3ed4040683b7339))
* **config:** alinhar limiares de seleção do deep learning ([7a0c47e](https://github.com/victorh-silveira/aether-quantum-engine/commit/7a0c47ecd0b42236ce11d01f4ed0ee8b9395f3d6))
* **config:** altera duracao padrao de trade para M30 e reativa todos os indices ([b610cf5](https://github.com/victorh-silveira/aether-quantum-engine/commit/b610cf5f08bdd367e52017341a62a09f2bf31460))
* **config:** altera o modo de rotulo para ma_trend e o horizonte do rotulo para 3 ([875ce3f](https://github.com/victorh-silveira/aether-quantum-engine/commit/875ce3f03d93b781ad150e47e0c4bcfce8928754))
* **config:** ativa logs detalhados LLM_IO na main ([7634162](https://github.com/victorh-silveira/aether-quantum-engine/commit/7634162eefa9ff1b997e255bcce0e0a35ba6669f))
* **config:** aumenta filtros de gating e ativa deploy gate para alta seletividade ([9afe3d5](https://github.com/victorh-silveira/aether-quantum-engine/commit/9afe3d587000c392c7fbeb5ae5615006ef009832))
* **config:** desativa operacao obrigatoria a cada ciclo para evitar perdas por baixa confianca ([13f3224](https://github.com/victorh-silveira/aether-quantum-engine/commit/13f3224d4a40e44d493af5448f1ecd3729b31bdb))
* **config:** desativar mhi_mode e expandir buffers ohlc ([8330312](https://github.com/victorh-silveira/aether-quantum-engine/commit/83303121dde98c1a33b1e6a12d9d78e0c0cd7579))
* **config:** elevar limites de conviccao minima para 85% no settings.json ([2ac5385](https://github.com/victorh-silveira/aether-quantum-engine/commit/2ac5385e301a830f03b06f89db3604731c02dc7c))
* **config:** elevar limites de conviccao minima para 85% no settings.json ([1bb6727](https://github.com/victorh-silveira/aether-quantum-engine/commit/1bb6727e72823140dac234097864eb3c70e10903))
* **config:** exclui indice OTC_FCHI por requerer duracao de M30 no broker ([bc2e424](https://github.com/victorh-silveira/aether-quantum-engine/commit/bc2e42420d4802892f6949f7c42ee7a78c59085a))
* **config:** exclui indice OTC_GDAXI por requerer duracao minima de M30 ([e77d797](https://github.com/victorh-silveira/aether-quantum-engine/commit/e77d797da0290c179ea8acfd8f01cf59ee1e8418))
* **config:** refina prompt medallion na main ([77c0651](https://github.com/victorh-silveira/aether-quantum-engine/commit/77c0651687e89591edbe758309aca357d4d464cc))
* **engine:** adicionar novos indicadores tecnicos ([cd5e849](https://github.com/victorh-silveira/aether-quantum-engine/commit/cd5e849e355429c56c3049c6aee8d1adc0a3d570))
* **engine:** adicionar novos indicadores tecnicos as features ([9e12297](https://github.com/victorh-silveira/aether-quantum-engine/commit/9e1229700105cf4d13cb8efbae95cc4e461a4895))
* **engine:** amplia amostragem DL, corrige boot e refatora motor live ([d25e7ec](https://github.com/victorh-silveira/aether-quantum-engine/commit/d25e7ec813eda5fc661f52d494164b23b8aee8b0))
* **engine:** consenso CALL/PUT unificado e limpeza do repositorio ([c7853de](https://github.com/victorh-silveira/aether-quantum-engine/commit/c7853de5c38738249845bb0af3de6ce66de76a73))
* **engine:** direcao DL guiada por fluxo de velas e mercado ([3d29435](https://github.com/victorh-silveira/aether-quantum-engine/commit/3d2943573d085e723821905ed2be17407aae8578))
* **engine:** fases de treino/operacao e execucao obrigatoria por ranking de mercado ([42b13d5](https://github.com/victorh-silveira/aether-quantum-engine/commit/42b13d5ffbc9c192d19cd67c70d2f56da91769e9))
* **engine:** filtros binarios CALL/PUT e simplificacao do pipeline ([191a201](https://github.com/victorh-silveira/aether-quantum-engine/commit/191a20156dc3fb830afcd4ce0c2332494e9c10ef))
* **engine:** M5 ma_trend e vol implicita para R_100 ([e6cafee](https://github.com/victorh-silveira/aether-quantum-engine/commit/e6cafeebd7ca1a3b165dc6f7d47350a0a4801637))
* **engine:** melhora features DL e treino M1 Rise/Fall ([fd1d333](https://github.com/victorh-silveira/aether-quantum-engine/commit/fd1d33304bead55a07d925e212b5162929e5427d))
* **engine:** otimizar hiperparâmetros e seletividade ([0ad912b](https://github.com/victorh-silveira/aether-quantum-engine/commit/0ad912b28a50c3202bc2bdd5b2a68897abe5566a))
* **engine:** Python 3.13.12, Conda deriv-api e refator DL/execucao ([5a01f89](https://github.com/victorh-silveira/aether-quantum-engine/commit/5a01f8990f5cdf538a0a1e4aad1f92f851994f9d))
* **engine:** refatora pipeline binario 60s e remove modulos legados ([f892c97](https://github.com/victorh-silveira/aether-quantum-engine/commit/f892c97fb073c480e806498d2ab901c8243f636b))
* **engine:** separa treino e execucao e reorganiza layout do repo ([4a80d62](https://github.com/victorh-silveira/aether-quantum-engine/commit/4a80d62f69ea012912298fa9260a8edb8469cc1a))
* **engine:** startup rapido em inferencia sem retreinar ([9d97763](https://github.com/victorh-silveira/aether-quantum-engine/commit/9d97763db115f63185a41a47c9f7f631a0e99f09))
* **engine:** treino por sessao, piso de score e logs estruturados ([9b09265](https://github.com/victorh-silveira/aether-quantum-engine/commit/9b09265918bcbbdc350748962234b513e3dceba2))
* **llm:** adiciona exemplo de resposta valida no system prompt ([6aef7c8](https://github.com/victorh-silveira/aether-quantum-engine/commit/6aef7c87f47718f5d5db639b02c46d9dab9b5dac))
* **llm:** adiciona trava de entropia e melhora prompt na main ([031a7f6](https://github.com/victorh-silveira/aether-quantum-engine/commit/031a7f6d414ad3bbde36b76e03e6d115892427e9))
* **llm:** aplica melhorias de prompt e parser na main ([8456def](https://github.com/victorh-silveira/aether-quantum-engine/commit/8456defc305f7eb6322a1c9e5e4e15e9a1b69c08))
* **llm:** atualiza modelo para gemini-3.1-pro-preview ([55eba17](https://github.com/victorh-silveira/aether-quantum-engine/commit/55eba17455a6b91957ed580674b9867c6b282ab6))
* **llm:** aumenta teto de probabilidade para 0.75 em mercados ruidosos na main ([b29e5a9](https://github.com/victorh-silveira/aether-quantum-engine/commit/b29e5a9e45b00601641ca4078ed63dd6c7de1383))
* **llm:** capacita o modelo a prever direcoes de cluster de forma independente ([5f06d54](https://github.com/victorh-silveira/aether-quantum-engine/commit/5f06d548715cdd106caa0fa75439901749a1caa7))
* **llm:** confluencia macro transatlantica com RISE/FALL e somente CALL/PUT ([0172ec5](https://github.com/victorh-silveira/aether-quantum-engine/commit/0172ec592a873f682bb34645bbca87593b629da5))
* **llm:** elevar conviccao maxima em divergencia macro ([e4f761e](https://github.com/victorh-silveira/aether-quantum-engine/commit/e4f761e3ddcdb491e0bbee728d6791ff607156a8))
* **llm:** execucao hibrida no cluster refresh e pilha quant M5 ([5153e2c](https://github.com/victorh-silveira/aether-quantum-engine/commit/5153e2c5dcca375308810cfd02e8403d6954d362))
* **llm:** implementa suporte a clusters US e EU ([e80a1cf](https://github.com/victorh-silveira/aether-quantum-engine/commit/e80a1cf3053d6167b00d822bc3ed4fe82a7cb543))
* **llm:** implementa suporte a clusters US e EU ([62f4951](https://github.com/victorh-silveira/aether-quantum-engine/commit/62f49513de6e81947a6cd475b892df2f64179d4e))
* **llm:** implementar motor Medallion StatArb e classificador de regime HMM ([d87cf6c](https://github.com/victorh-silveira/aether-quantum-engine/commit/d87cf6c4f070c0c4fc6194c6c88836a02eaae873))
* **llm:** injeta metricas de indice de cluster em tempo real no prompt sniper ([4dc2a4e](https://github.com/victorh-silveira/aether-quantum-engine/commit/4dc2a4e95f143b8ab1f7ca43cc4840d8f0f3ee79))
* **llm:** inversao de cluster e execucao exclusiva por macro ([135b43e](https://github.com/victorh-silveira/aether-quantum-engine/commit/135b43e633ad9869e5c91926fc1a289e3c9bb97e))
* **llm:** Medallion Gemini tag_change, filtros macro e backtest assertivo ([cd21cd2](https://github.com/victorh-silveira/aether-quantum-engine/commit/cd21cd2973cb2b11c8ef08edd0acaef0612475de))
* **llm:** modo inteligencia macro pura estilo Medallion ([556034a](https://github.com/victorh-silveira/aether-quantum-engine/commit/556034a79728a54f4246e34c85d88f35d63f4659))
* **llm:** otimizar sinteticos M1 pos-win e log de inversao ([27d79e2](https://github.com/victorh-silveira/aether-quantum-engine/commit/27d79e22af7257c17e35262e47458c1f0dcb8c57))
* **llm:** prompt com seis timeframes e propagacao por cluster ([21ea949](https://github.com/victorh-silveira/aether-quantum-engine/commit/21ea949ac1fb068c90870226431b57d011453009))
* **llm:** reduz limite de entropia para 3.0 e trava para 0.69 na main ([3479a3e](https://github.com/victorh-silveira/aether-quantum-engine/commit/3479a3ec764a421cbb297cd75a444110688d71f3))
* **llm:** refatorar motor Medallion puro com propagacao regional e StatArb ([72fb474](https://github.com/victorh-silveira/aether-quantum-engine/commit/72fb4745b9e3820bbdc1f4985a8c77991dcbd3bd))
* **llm:** remove SSMI do cluster e adiciona regras de trading no prompt ([95c67df](https://github.com/victorh-silveira/aether-quantum-engine/commit/95c67dfcb213bac1ed60a26922fd146b317723b2))
* migração para 100% deep learning PyTorch e recuperação Martingale cross-symbol em mercado único ([2f273c6](https://github.com/victorh-silveira/aether-quantum-engine/commit/2f273c6a4d582af8fba884556c8eb5f4570b9198))
* **orchestrator:** alinhar StatArb risk_on e fallback de indices ([7155ba1](https://github.com/victorh-silveira/aether-quantum-engine/commit/7155ba17bcd165cb652bdfe3e240e1a14b07f3e0))
* **orchestrator:** backtest walk-forward e filtros de cenario lucrativo ([f4473b9](https://github.com/victorh-silveira/aether-quantum-engine/commit/f4473b9c7cbfaa2e5981160615ba29e0abc6f259))
* **orchestrator:** ciclo pos-liquidacao, refresh StatArb e inversao ([59d5c20](https://github.com/victorh-silveira/aether-quantum-engine/commit/59d5c2042df21fff11b9ece2803d72d6493baa78))
* **orchestrator:** desacopla clusters e aprimora liquidacao ([7a8df85](https://github.com/victorh-silveira/aether-quantum-engine/commit/7a8df85985b86a0de5ab0a32b9032a5380ad2adb))
* **orchestrator:** folego pos-liquidacao e inversao segura ([d56c00d](https://github.com/victorh-silveira/aether-quantum-engine/commit/d56c00ddc2c2b0e0a5bbe85e7e39c392043eeeca))
* **orchestrator:** quarentena de inversao apos loss ([ee2fb32](https://github.com/victorh-silveira/aether-quantum-engine/commit/ee2fb3227fcf08c39465a33d0264af5df07a924b))
* **orchestrator:** remove codigo morto e arquivo nao utilizado ([0a05d17](https://github.com/victorh-silveira/aether-quantum-engine/commit/0a05d170d9c3f4f2038a6dccc27f9da8f90d7ce3))
* **orchestrator:** resolve rate limit de proposta e otimiza clusters ([779da44](https://github.com/victorh-silveira/aether-quantum-engine/commit/779da44c1f720cb95f4a766fdc7d0fb90d3194d3))
* **repo:** commit inicial do projeto ([0d35232](https://github.com/victorh-silveira/aether-quantum-engine/commit/0d3523284600179317b157321d74563287d2cbf8))
* **repo:** remove arquivo de changelog ([e1491a0](https://github.com/victorh-silveira/aether-quantum-engine/commit/e1491a03a6a04ee203a066c9d3dc92f7268d5906))
* **risk-statarb:** otimizacao do medallion, remocao de limites de kelly/recuperacao e ativacao de clusters transatlanticos ([5d93ae2](https://github.com/victorh-silveira/aether-quantum-engine/commit/5d93ae26e0c35f92fb9dfb5b66e7d9270df9cb18))
* **risk:** adiciona controle de perdas consecutivas e escalonamento de cooldown ([0f7c5bf](https://github.com/victorh-silveira/aether-quantum-engine/commit/0f7c5bf5fe33414d6cbec6a595257e2eeea239d3))
* **risk:** aumenta limiares de confianca e score minimo de execucao ([78842f6](https://github.com/victorh-silveira/aether-quantum-engine/commit/78842f6f2c1f917013e87e2f6946ada335a83dcd))
* **risk:** implementa logica cirurgica do single strike e volatilidade dinamica HMM ([13ece6f](https://github.com/victorh-silveira/aether-quantum-engine/commit/13ece6f6288035255adc117c3a9dd48f051fe02d))
* **risk:** otimiza modelo de entrada unica para stop win em 30m e ajusta prompts e inversao ([9807dd2](https://github.com/victorh-silveira/aether-quantum-engine/commit/9807dd2bd7937cd11630d30c497cf66fe5b24cb5))
* **risk:** otimizar parametros do Medallion para recuperacao, frequencia e escopo de cluster ([a0a7683](https://github.com/victorh-silveira/aether-quantum-engine/commit/a0a7683f9e1fcd9d696fa9e8d1302047ba9e5b8d))
* **risk:** reduz fracao de lucro alvo do martingale para zero ([35d5380](https://github.com/victorh-silveira/aether-quantum-engine/commit/35d53809ed73de603d4a2b04324b36893e5c5560))
* **risk:** refina gerenciamento de risco e logica de inversao de direcao na recuperacao ([8f63d2a](https://github.com/victorh-silveira/aether-quantum-engine/commit/8f63d2a01307c13e407145f2983cda5bb21c7d26))
* **risk:** remover limites de martingale e travas de stake no settings ([ca72f84](https://github.com/victorh-silveira/aether-quantum-engine/commit/ca72f84f8c4b46359f8dbf58f3eaab7f806a27a7))
* **risk:** sessao OTC, stake tier e refresh LLM periodico ([0388e33](https://github.com/victorh-silveira/aether-quantum-engine/commit/0388e331858f04568b070a33300764a75d1a5ad7))
* **scripts:** adiciona pylint para deteccao nativa de codigo duplicado ([1b1d437](https://github.com/victorh-silveira/aether-quantum-engine/commit/1b1d437d0e2f89a9642917a8f11fd4ba759ff2aa))

### Correcoes de Bug

* **app:** forca resolve_startup_fetch_bars a carregar historico completo no modo de treino ([2b87c31](https://github.com/victorh-silveira/aether-quantum-engine/commit/2b87c317caa687a18eb4b343041fd5589031d2bb))
* **app:** limitar threads de cpu do pytorch para evitar deadlock assincrono ([6d20821](https://github.com/victorh-silveira/aether-quantum-engine/commit/6d208216adaf0648758128d2a25477bf0faeb034))
* **config:** aumentar velas de inicializacao de 512 para 1024 ([71f5256](https://github.com/victorh-silveira/aether-quantum-engine/commit/71f52566c87e0b4f92d69ce2d36d97b24cf0f609))
* **config:** corrige erro de decode no windows e ativa modo live ([189777f](https://github.com/victorh-silveira/aether-quantum-engine/commit/189777f46900adf82fe416e1ab145aae79c2b0f3))
* **dl:** corrige path do mock no teste de coleta de deep learning ([972700b](https://github.com/victorh-silveira/aether-quantum-engine/commit/972700baaa727ab76232cef0ad24a3cca861b263))
* **engine:** alinhar clusters ao macro e corrigir liquidacao e persistencia ([e892f23](https://github.com/victorh-silveira/aether-quantum-engine/commit/e892f23759a800cf728dd1c7a671ba12f2d8ffcf))
* **engine:** aumenta Kelly, martingale integral e inverte direcao no recovery ([357331d](https://github.com/victorh-silveira/aether-quantum-engine/commit/357331d0b535488040afb23f459df07fc9fea9ea))
* **engine:** corrige bootstrap DL no run.py e acelera inferencia ([5542856](https://github.com/victorh-silveira/aether-quantum-engine/commit/55428565367b4e615b4adc396b675954cd6e9069))
* **engine:** corrigir amostragem no redimensionamento de M1 para M5 ([f44b194](https://github.com/victorh-silveira/aether-quantum-engine/commit/f44b194755eb3bbac08b2210c898d47f73a643e6))
* **engine:** desbloqueia trades pos-loss, recalibra Kelly M5 e reduz spam de liquidacao ([7c2bcf4](https://github.com/victorh-silveira/aether-quantum-engine/commit/7c2bcf416321c5eb8c8f79b9d8a53c601a70aef8))
* **engine:** detectar checkpoints de feature_dim incompativel no startup ([c7e00c4](https://github.com/victorh-silveira/aether-quantum-engine/commit/c7e00c46abfa69cfc9f880c8874d654e7e53ce25))
* **engine:** fortalecer recovery obrigatório e sizing sem teto ([19921a6](https://github.com/victorh-silveira/aether-quantum-engine/commit/19921a672c43a3f081c5ec1247e75baa46d243eb))
* **engine:** Kelly conservador, liquidacao resiliente e reconexao WS ([e32f5d9](https://github.com/victorh-silveira/aether-quantum-engine/commit/e32f5d9330366784b64cadd03ff13de09c82eeb0))
* **engine:** opera M1 com modelo M5 sem retreino e endurece gates ([c7d040b](https://github.com/victorh-silveira/aether-quantum-engine/commit/c7d040bb436625a7e8a87a872f9ffdd962a07322))
* **engine:** protege direcao DL forte e ranqueia por score no modo obrigatorio ([91f3f68](https://github.com/victorh-silveira/aether-quantum-engine/commit/91f3f68302b1773130fb1729780046277df1a196))
* **engine:** recalculacao dinamica de grace period ([bbb08c7](https://github.com/victorh-silveira/aether-quantum-engine/commit/bbb08c7a1f9ceda7b10a71c9c2d789326b883009))
* **llm:** aceitar conviccao LLM em divergencia sem piso por tag ([bcba776](https://github.com/victorh-silveira/aether-quantum-engine/commit/bcba77621599b095ac186d3a8edc0d46dc6af77a))
* **llm:** desativa thinking e corrige truncamento MAX_TOKENS ([1732029](https://github.com/victorh-silveira/aether-quantum-engine/commit/17320294403d7e17aba6ae4227ee1d10023c1c9c))
* **llm:** ignorar ruido US em divergencia e pular cluster flat ([90cce65](https://github.com/victorh-silveira/aether-quantum-engine/commit/90cce654e864d7624e91e0cad23c95bc5f8ae851))
* **llm:** inverter lado LLM e contratos M1 nos sinteticos ([ebd1b3d](https://github.com/victorh-silveira/aether-quantum-engine/commit/ebd1b3d9d23d76aadcff1029ff30de39c283030e))
* **llm:** isolamento do parser de decisoes e garantia de cobertura ([21652b4](https://github.com/victorh-silveira/aether-quantum-engine/commit/21652b453cbbe01a681da64934bca73003d9a88e))
* **llm:** liberar trades apos loss com cluster refresh risk_off ([37618e6](https://github.com/victorh-silveira/aether-quantum-engine/commit/37618e697b66af93b85cb747dce43a1fdbef46a3))
* **llm:** manter Gemini ativo em indefinido e sem pausa pos-loss ([0e4dd86](https://github.com/victorh-silveira/aether-quantum-engine/commit/0e4dd86c9a7051a3ec2bd796b6180f66e3fe4570))
* **llm:** preservar direcao Gemini em tags de divergencia ([f933e37](https://github.com/victorh-silveira/aether-quantum-engine/commit/f933e371002adc853139691fc0a1fea85162af05))
* **llm:** propagacao por indice, parser US/EU compacto e log LLM_IO off ([602f0d4](https://github.com/victorh-silveira/aether-quantum-engine/commit/602f0d4e663e1c7d7cccf9239d09a0cf7d01998e))
* **llm:** reduz timeouts Gemini com flash, menos retries e fallback ([ec02c6e](https://github.com/victorh-silveira/aether-quantum-engine/commit/ec02c6eeaf7e0c822bcd757e99814f1ac7003860))
* **llm:** reforcar vetos e penalidades em divergencia macro ([a1ae2dd](https://github.com/victorh-silveira/aether-quantum-engine/commit/a1ae2ddfdf54387498ff3d6c8c215fdd6a53cc63))
* **llm:** remove fallback de indices e exige CALL/PUT somente da LLM ([69d1396](https://github.com/victorh-silveira/aether-quantum-engine/commit/69d1396aa4473e51b2f8cb91bedb3b1285f82fce))
* **llm:** saida JSON obrigatoria CALL/PUT e fallback flash-lite ([9bafe69](https://github.com/victorh-silveira/aether-quantum-engine/commit/9bafe692672391b8200ef8d6f8c4fba3bad99c5a))
* **llm:** trocar vetos HMM por inteligencia e liberar divergencia LLM ([c062855](https://github.com/victorh-silveira/aether-quantum-engine/commit/c0628554371625b65fc8bc80a9f868736c3b2fd9))
* **orchestrator:** agendar proximo ciclo de decisao quando nenhuma ordem e executada ([b57bca4](https://github.com/victorh-silveira/aether-quantum-engine/commit/b57bca42fdc635bffd0e46e3a319c2d0ce343132))
* **orchestrator:** corrigir exec hold e processamento duplicado no mesmo candle ([566c077](https://github.com/victorh-silveira/aether-quantum-engine/commit/566c0777009234f0d9ff2c25495d8ece4af05cb0))
* **orchestrator:** descartar candidatos execute=False durante recovery ([60d7e85](https://github.com/victorh-silveira/aether-quantum-engine/commit/60d7e8541fff30f0f984b7c34f82be3acbc3f014))
* **orchestrator:** garantir recovery obrigatorio sem EXEC_SKIP ([8546680](https://github.com/victorh-silveira/aether-quantum-engine/commit/8546680ddcc9449d3c40999b333d2f35f4c4504d))
* **orchestrator:** move set_trading(False) para dentro do bloco de stop win na main ([9ff1af2](https://github.com/victorh-silveira/aether-quantum-engine/commit/9ff1af2ed29cb81c21258309f35b83fbaac1b75d))
* **orchestrator:** respeita a granularidade configurada em modo treino ([0a95b6e](https://github.com/victorh-silveira/aether-quantum-engine/commit/0a95b6ecba8218f94a0895f4e8670077bd63e11f))
* **orchestrator:** retomar ciclo apos liquidacao sem congelar o motor ([869305f](https://github.com/victorh-silveira/aether-quantum-engine/commit/869305f308942026ff6f06dc1f622434551bb20b))
* **risk:** martingale dobra ultima entrada e setup WSL ([335069b](https://github.com/victorh-silveira/aether-quantum-engine/commit/335069b9c82a24e90f5e81c9b10c52c976bd028b))
* **risk:** nao injetar ou forcar par de recovery se execute=False ([19f92eb](https://github.com/victorh-silveira/aether-quantum-engine/commit/19f92eb4eeb93d7c1fe4a934eb27875af6a10872))
* **risk:** remover teto de martingale e alinhar direcao de recovery ([a1fb89a](https://github.com/victorh-silveira/aether-quantum-engine/commit/a1fb89ac6841b4ad6136cad645a8a88b49afeff7))
* **risk:** suavizar Kelly e desativar stop-win agressivo ([0f1cec5](https://github.com/victorh-silveira/aether-quantum-engine/commit/0f1cec5acb7aac301145e613c12f5b2a591d5c5f))
* **scripts:** ajustar atalho do run.py nos scripts de launch ([3658c0b](https://github.com/victorh-silveira/aether-quantum-engine/commit/3658c0b8b7af5dbb2b65caa9013e221547fb714c))
* **scripts:** atualiza script de ping para gemini-3.1-pro-preview ([17d2c27](https://github.com/victorh-silveira/aether-quantum-engine/commit/17d2c27d9458e9167325e1b78deccea51297a9d0))
* **scripts:** resolve python do venv do projeto para quality gates ([c6457fd](https://github.com/victorh-silveira/aether-quantum-engine/commit/c6457fd5d9e251b5215e90aab4d0264929389559))
* **test:** adiciona cobertura para o load_dotenv no auth_manager ([65c7a9a](https://github.com/victorh-silveira/aether-quantum-engine/commit/65c7a9a5a56d9a140788490c71468b237ea32bb7))
* **test:** adiciona teste para early stopping e corrige resolve_conda_python ([def229b](https://github.com/victorh-silveira/aether-quantum-engine/commit/def229b844945b512b80fbfd4e59e76dbf6d3fd7))
* **tools:** hooks WSL, refator lint e docstrings Medallion ([a844879](https://github.com/victorh-silveira/aether-quantum-engine/commit/a84487932d4c14b8f5cf91bcaeb29e170d9a5177))

### Refatoracoes Tecnicas

* **app:** reorganiza indicadores e testes para limite de 300 linhas ([0d4e126](https://github.com/victorh-silveira/aether-quantum-engine/commit/0d4e1260b57bc8c0fb7178e9fbd723de1cb2db43))
* **backtest:** remove completamente o sistema de backtest offline e documentacao ([a415e48](https://github.com/victorh-silveira/aether-quantum-engine/commit/a415e4895615b6730ffc4df9430124b3ebb4fd5f))
* **config:** adiciona filtro m5 e aumenta thresholds ([715de4f](https://github.com/victorh-silveira/aether-quantum-engine/commit/715de4f12b19cb4bd20e80fbd7290bf316541911))
* **config:** atualiza versao do Python para 3.14.5 ([ae849f6](https://github.com/victorh-silveira/aether-quantum-engine/commit/ae849f68a868cbc0ad98269af8b4def8ea6c37cc))
* **config:** expande as verificacoes de qualidade para todos os arquivos python ([6f7f2ac](https://github.com/victorh-silveira/aether-quantum-engine/commit/6f7f2ac0c69ae81f612a6700f825aa180c699457))
* **config:** remove nomes ignorados do vulture ([2c50585](https://github.com/victorh-silveira/aether-quantum-engine/commit/2c50585d42ded74533aecb2437c1defa10c695b0))
* **llm:** refina diretrizes quant do system_prompt para reversao de zscore extremo ([ac8cd53](https://github.com/victorh-silveira/aether-quantum-engine/commit/ac8cd53ddd394e8773f496512c4800a779be9bdb))
* **llm:** remove chaves de config inativas e adiciona verificacoes de websocket ([9e7ea28](https://github.com/victorh-silveira/aether-quantum-engine/commit/9e7ea2809ad8660d18a7976c1fb12e9ed4eb470f))
* **llm:** remove fallback de execucao e aprimora prompt do gemini ([d2108fe](https://github.com/victorh-silveira/aether-quantum-engine/commit/d2108febc2585e046accf6cf94090269f5b4c7cd))
* **orchestrator:** otimiza camadas de software e ajusta conformidade do codigo ([98ca843](https://github.com/victorh-silveira/aether-quantum-engine/commit/98ca843e0a2ca135bf9b15279489e3485ff4cb28))
* **repo:** layout app, docs e linters sem infra K8s ([df36e43](https://github.com/victorh-silveira/aether-quantum-engine/commit/df36e43d9fafa25a90200d4ae04ab8ca38b12cb5))
* **repo:** preservar o diretorio de dados no cleanup ([9fe1d8f](https://github.com/victorh-silveira/aether-quantum-engine/commit/9fe1d8fe5d4ede7c261d6d59f6e1ff553c73caf8))
* **test:** atualiza suite de testes unitarios para simbolos ativos ([8b0234b](https://github.com/victorh-silveira/aether-quantum-engine/commit/8b0234b1c4b48481a4921fbfa0979d0d5ec265e8))

### Documentacao

* **deriv:** adiciona documentacao do algoritmo CSPRNG da Deriv e corrige referencias a simbolos legados ([dbb1312](https://github.com/victorh-silveira/aether-quantum-engine/commit/dbb1312bd2c4631a5f958e39810c5a615bdd1042))
* indicar branch sinteticos no README main ([89cb74d](https://github.com/victorh-silveira/aether-quantum-engine/commit/89cb74dafcc9894f220179db112b290551651e1b))
* **release:** remover secao Unreleased do CHANGELOG ([f97f862](https://github.com/victorh-silveira/aether-quantum-engine/commit/f97f862dcc56f533aa3cbf94f8d0e4127316af49))
* remove referencias a branch de indices sinteticos ([3557d6f](https://github.com/victorh-silveira/aether-quantum-engine/commit/3557d6f29280f988e0ea545520309f5b1297bd6f))
* **repo:** atualiza README.md para modelo medallion na main ([8c56109](https://github.com/victorh-silveira/aether-quantum-engine/commit/8c56109b74cbec6506b65f1e819353e0b8dd529e))
* **repo:** mantem apenas tags existentes no changelog ([48e1689](https://github.com/victorh-silveira/aether-quantum-engine/commit/48e1689f8f1f610452fb6fb50129e65c719ab44e))

## [1.20.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.19.0...v1.20.0) (2026-06-19)

### Funcionalidades

* **app:** adiciona filtro de tendencia curto configuravel ([e94b130](https://github.com/victorh-silveira/aether-quantum-engine/commit/e94b1301f9df44fbbb8680c9f1b7235be5b05f36))
* **app:** adiciona inclinacao no calculo de tendencia ([f147dcb](https://github.com/victorh-silveira/aether-quantum-engine/commit/f147dcbb6b4ec63506e57c2d152db044320f6e2c))

## [1.19.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.18.1...v1.19.0) (2026-06-18)

### Funcionalidades

* **app:** adiciona confirmacao de tendencia SMA-20 no flip de recuperacao ([76bbdd4](https://github.com/victorh-silveira/aether-quantum-engine/commit/76bbdd4015257a3d8075fdacc2cc588202e97c9a))
* **app:** adiciona fallback de tendencia SMA-20 em execucao ([ad9e3f5](https://github.com/victorh-silveira/aether-quantum-engine/commit/ad9e3f5df5f11f0d8d493630dcc555acb8351a3e))
* **app:** remove bloqueios de cooldown e session pause para eliminar holds ([94b525b](https://github.com/victorh-silveira/aether-quantum-engine/commit/94b525b70fbec64f7cbbae1091f169382621265a))
* **risk:** aumenta limiares de confianca e score minimo de execucao ([79410f5](https://github.com/victorh-silveira/aether-quantum-engine/commit/79410f56b00bad0a0a57cfab56d99d8ed8a1d737))
* **risk:** reduz fracao de lucro alvo do martingale para zero ([1c3f11b](https://github.com/victorh-silveira/aether-quantum-engine/commit/1c3f11b2c84ae7393088d7bf1429ba5186a39d38))
* **risk:** refina gerenciamento de risco e logica de inversao de direcao na recuperacao ([aabe2c6](https://github.com/victorh-silveira/aether-quantum-engine/commit/aabe2c62fb15d88569566d3ddedfb8c23b364f21))

## [1.18.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.18.0...v1.18.1) (2026-06-18)

### Correcoes de Bug

* **orchestrator:** corrigir exec hold e processamento duplicado no mesmo candle ([ad70042](https://github.com/victorh-silveira/aether-quantum-engine/commit/ad70042281029eedf0a3e7753d9e397172ab7ea1))

## [1.18.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.17.1...v1.18.0) (2026-06-18)

### Funcionalidades

* **risk:** remover limites de martingale e travas de stake no settings ([083b962](https://github.com/victorh-silveira/aether-quantum-engine/commit/083b962dbd796fdc6c57cc4fc678321350e53c2a))

### Correcoes de Bug

* **engine:** recalculacao dinamica de grace period ([9d7cf80](https://github.com/victorh-silveira/aether-quantum-engine/commit/9d7cf805595cb2a164f0ccf8682a2eb086b764a3))
* **orchestrator:** agendar proximo ciclo de decisao quando nenhuma ordem e executada ([55b2b99](https://github.com/victorh-silveira/aether-quantum-engine/commit/55b2b992d5c13e6e7cea72423390400f883102ce))
* **orchestrator:** respeita a granularidade configurada em modo treino ([f910513](https://github.com/victorh-silveira/aether-quantum-engine/commit/f9105133998ecc1c50dadbb45ab94c93aac64dfc))

### Refatoracoes Tecnicas

* **app:** reorganiza indicadores e testes para limite de 300 linhas ([d32399a](https://github.com/victorh-silveira/aether-quantum-engine/commit/d32399a3cb8419f2bf2ea45d7fdf5168f876baf2))
* **orchestrator:** otimiza camadas de software e ajusta conformidade do codigo ([c9c3392](https://github.com/victorh-silveira/aether-quantum-engine/commit/c9c33927896d9543e0b73d3cf6b3f983dc0a6341))

## [1.17.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.17.0...v1.17.1) (2026-06-17)

### Correcoes de Bug

* **engine:** detectar checkpoints de feature_dim incompativel no startup ([959932a](https://github.com/victorh-silveira/aether-quantum-engine/commit/959932ad49f39ee26dadfbb828294e08fee76d3c))

## [1.17.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.16.0...v1.17.0) (2026-06-17)

### Funcionalidades

* **engine:** adicionar novos indicadores tecnicos ([95b8413](https://github.com/victorh-silveira/aether-quantum-engine/commit/95b8413337baef7fcd312345e433638d03a8a95e))

## [1.16.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.15.2...v1.16.0) (2026-06-16)

### Funcionalidades

* **engine:** adicionar novos indicadores tecnicos as features ([f51073c](https://github.com/victorh-silveira/aether-quantum-engine/commit/f51073c40e419f1d4e4d2162d4ed78c53d5428e8))

### Refatoracoes Tecnicas

* **repo:** preservar o diretorio de dados no cleanup ([d65e31c](https://github.com/victorh-silveira/aether-quantum-engine/commit/d65e31ca5bc629b46910bda06f73f380d29d7cde))

## [1.15.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.15.1...v1.15.2) (2026-06-16)

### Correcoes de Bug

* **config:** aumentar velas de inicializacao de 512 para 1024 ([ca2f4ad](https://github.com/victorh-silveira/aether-quantum-engine/commit/ca2f4ad8ebe791a82e99b54352da42c2529fb4a7))

## [1.15.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.15.0...v1.15.1) (2026-06-16)

### Correcoes de Bug

* **engine:** corrigir amostragem no redimensionamento de M1 para M5 ([7a36089](https://github.com/victorh-silveira/aether-quantum-engine/commit/7a360890a10b1ea8f3c80bdd893ca8be6eb6571f))

## [1.15.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.14.1...v1.15.0) (2026-06-16)

### Funcionalidades

* **config:** alinhar limiares de seleção do deep learning ([f4755fe](https://github.com/victorh-silveira/aether-quantum-engine/commit/f4755feaf856ea5c2f364c6f5baa2d78897924c3))
* **engine:** otimizar hiperparâmetros e seletividade ([2b24396](https://github.com/victorh-silveira/aether-quantum-engine/commit/2b24396a7f1c5c2c00821fd17e0b03286eb0bca5))

## [1.15.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.14.1...v1.15.0) (2026-06-16)

### Funcionalidades

* **engine:** optimize hyperparameters and selectivity ([721b6c7](https://github.com/victorh-silveira/aether-quantum-engine/commit/721b6c7908c969a921e2f93c0681c247676a573f))

## [1.14.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.14.0...v1.14.1) (2026-06-16)

### Correcoes de Bug

* **engine:** fortalecer recovery obrigatório e sizing sem teto ([b831c4a](https://github.com/victorh-silveira/aether-quantum-engine/commit/b831c4a9beb0289136fe393d589d6f9f3bee468b))

## [1.14.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.13.4...v1.14.0) (2026-06-15)

### Funcionalidades

* **engine:** startup rapido em inferencia sem retreinar ([40f605a](https://github.com/victorh-silveira/aether-quantum-engine/commit/40f605a89c710d2a1b6ca6c35a91b1e43697e2e3))

## [1.13.4](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.13.3...v1.13.4) (2026-06-15)

### Correcoes de Bug

* **engine:** opera M1 com modelo M5 sem retreino e endurece gates ([c3d13ab](https://github.com/victorh-silveira/aether-quantum-engine/commit/c3d13abe8e8e78993c4d997d1334225d7f34c4da))

## [1.13.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.13.2...v1.13.3) (2026-06-15)

### Correcoes de Bug

* **engine:** aumenta Kelly, martingale integral e inverte direcao no recovery ([b7e8348](https://github.com/victorh-silveira/aether-quantum-engine/commit/b7e8348cc11aafb7943a7a1cfe8b700b929321a2))

## [1.13.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.13.1...v1.13.2) (2026-06-15)

### Correcoes de Bug

* **engine:** desbloqueia trades pos-loss, recalibra Kelly M5 e reduz spam de liquidacao ([d47ff2b](https://github.com/victorh-silveira/aether-quantum-engine/commit/d47ff2bc45693105198ee28fd829c6fad35676c3))

## [1.13.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.13.0...v1.13.1) (2026-06-15)

### Correcoes de Bug

* **engine:** corrige bootstrap DL no run.py e acelera inferencia ([f9a758d](https://github.com/victorh-silveira/aether-quantum-engine/commit/f9a758d0166378d8b723d9d9614e4e5d1664937e))

## [1.13.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.12.0...v1.13.0) (2026-06-14)

### Funcionalidades

* **engine:** M5 ma_trend e vol implicita para R_100 ([720461c](https://github.com/victorh-silveira/aether-quantum-engine/commit/720461c9f984ab7988fee6b90f9b0b1ed963b97a))

## [1.12.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.11.0...v1.12.0) (2026-06-14)

### Funcionalidades

* **engine:** separa treino e execucao e reorganiza layout do repo ([de1494d](https://github.com/victorh-silveira/aether-quantum-engine/commit/de1494dc5d6b33a9ac264c17d1b3b9a8b9421be6))

## [1.11.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.10.0...v1.11.0) (2026-06-14)

### Funcionalidades

* **engine:** melhora features DL e treino M1 Rise/Fall ([0d63878](https://github.com/victorh-silveira/aether-quantum-engine/commit/0d63878ccd9823e52712c247a973ef838e6679b8))

## [1.10.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.9.0...v1.10.0) (2026-06-14)

### Funcionalidades

* **engine:** refatora pipeline binario 60s e remove modulos legados ([5aa6830](https://github.com/victorh-silveira/aether-quantum-engine/commit/5aa6830836fcbe42bd927683f822d4f9d8546285))

## [Unreleased]

### Breaking Changes

* **engine:** refatoracao completa Deep Learning — metodologia Rise/Fall
  * Removida estrategia TREND_FIBO e modulos DL legados (consensus, regime, pair features, binary_signal)
  * Labels binarios alinhados ao contrato 60 s (horizon = 1 barra)
  * 19 features: log-return, RSI/delta-RSI, BB %B, ATR, distancia EMA20/50, ROC, microestrutura, Hurst, volatilidade
  * Lookback padrao 30 barras M1; historico de treino 130k velas (~3 meses)
  * Early stopping por validation loss (max 50 epocas, patience 6)
  * Arquiteturas TCN (padrao), LSTM e GRU configuravel via `deep_learning.arch`
  * Gating simplificado: threshold 0.75 CALL / 0.25 PUT (abstencao no meio)
  * `mandatory_trade_each_cycle: false` — operacao seletiva
  * TorchScript (`_ts.pt`) para inferencia rapida
  * Checkpoints v4 invalidam modelos anteriores

## [1.9.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.8.1...v1.9.0) (2026-06-11)

### Funcionalidades

* **engine:** consenso CALL/PUT unificado e limpeza do repositorio ([cdfee24](https://github.com/victorh-silveira/aether-quantum-engine/commit/cdfee241ffe5b652a04eb46d01c11b4c16451b4a))

## [1.8.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.8.0...v1.8.1) (2026-06-11)

### Correcoes de Bug

* **engine:** protege direcao DL forte e ranqueia por score no modo obrigatorio ([0db7825](https://github.com/victorh-silveira/aether-quantum-engine/commit/0db7825416524d1d9a8d192e1e4bb4c08ae28b4d))

## [1.8.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.7.0...v1.8.0) (2026-06-10)

### Funcionalidades

* **engine:** direcao DL guiada por fluxo de velas e mercado ([720bbcf](https://github.com/victorh-silveira/aether-quantum-engine/commit/720bbcfd24de56e5f8ee3d7600f6b1ce016f0e7c))

## [1.7.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.6.0...v1.7.0) (2026-06-10)

### Funcionalidades

* **engine:** treino por sessao, piso de score e logs estruturados ([7796548](https://github.com/victorh-silveira/aether-quantum-engine/commit/779654821583bf59f2e67b3648a3dd12512dc552))

## [1.6.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.5.2...v1.6.0) (2026-06-10)

### Funcionalidades

* **engine:** fases de treino/operacao e execucao obrigatoria por ranking de mercado ([0f670a3](https://github.com/victorh-silveira/aether-quantum-engine/commit/0f670a3570fb40b1b9c66494db1cd2abf22ba9a7))

## [1.5.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.5.1...v1.5.2) (2026-06-10)

### Correcoes de Bug

* **engine:** Kelly conservador, liquidacao resiliente e reconexao WS ([44d0988](https://github.com/victorh-silveira/aether-quantum-engine/commit/44d09882e11e3a58e0edbacbef62344d2fd3d20d))

## [1.5.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.5.0...v1.5.1) (2026-06-09)

### Correcoes de Bug

* **orchestrator:** garantir recovery obrigatorio sem EXEC_SKIP ([82ff00b](https://github.com/victorh-silveira/aether-quantum-engine/commit/82ff00bfb26d3f764e7c313e567dab01ab1e1c9c))

## [1.5.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.4.2...v1.5.0) (2026-06-09)

### Funcionalidades

* **engine:** filtros binarios CALL/PUT e simplificacao do pipeline ([5c23269](https://github.com/victorh-silveira/aether-quantum-engine/commit/5c232699e51edceb6673e77afd3731dfeabe6000))

## [1.4.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.4.1...v1.4.2) (2026-06-09)

### Correcoes de Bug

* **orchestrator:** retomar ciclo apos liquidacao sem congelar o motor ([25ddb84](https://github.com/victorh-silveira/aether-quantum-engine/commit/25ddb84b0c9e74f91ac09530c9cfc2852d6ac666))

## [1.4.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.4.0...v1.4.1) (2026-06-08)

### Correcoes de Bug

* **app:** limitar threads de cpu do pytorch para evitar deadlock assincrono ([1808e01](https://github.com/victorh-silveira/aether-quantum-engine/commit/1808e01afc8619f71dbad10039873710a2e95713))

## [1.4.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.3.0...v1.4.0) (2026-06-08)

### Funcionalidades

* **config:** desativar mhi_mode e expandir buffers ohlc ([f4dcc93](https://github.com/victorh-silveira/aether-quantum-engine/commit/f4dcc930d80e2ef7d6c44cab025c34d1a81dbb33))

## [1.3.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.2.0...v1.3.0) (2026-06-08)

### Funcionalidades

* **app:** resetar saldo demo automaticamente se zerado ([7184869](https://github.com/victorh-silveira/aether-quantum-engine/commit/7184869b9025a5c5eaf3709263e090d4d122aed5))

## [1.2.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.1.3...v1.2.0) (2026-06-08)

### Funcionalidades

* **app:** ajustar direcao em recovery e adicionar filtro rsi ([7dc8e3d](https://github.com/victorh-silveira/aether-quantum-engine/commit/7dc8e3d8c8f07fc5e40619e84def3e114aef9067))

## [1.1.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.1.2...v1.1.3) (2026-06-07)

### Correcoes de Bug

* **risk:** nao injetar ou forcar par de recovery se execute=False ([c7b6a99](https://github.com/victorh-silveira/aether-quantum-engine/commit/c7b6a9981dd025d77637397d9071fc29b59d597a))

## [1.1.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.1.1...v1.1.2) (2026-06-07)

### Correcoes de Bug

* **orchestrator:** descartar candidatos execute=False durante recovery ([715d949](https://github.com/victorh-silveira/aether-quantum-engine/commit/715d94924fa9b82e591d2907a21d372a214e2de9))

## [1.1.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.1.0...v1.1.1) (2026-06-07)

### Correcoes de Bug

* **risk:** remover teto de martingale e alinhar direcao de recovery ([de97a25](https://github.com/victorh-silveira/aether-quantum-engine/commit/de97a25e807f1635f39980ed478c55d8b69681ea))

## [1.1.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.0.0...v1.1.0) (2026-06-05)

### Funcionalidades

* **config:** aumenta filtros de gating e ativa deploy gate para alta seletividade ([39ff06c](https://github.com/victorh-silveira/aether-quantum-engine/commit/39ff06cf09f1521aed1fb7d6669a1aa782e2d30b))

### Documentacao

* **repo:** mantem apenas tags existentes no changelog ([3413305](https://github.com/victorh-silveira/aether-quantum-engine/commit/3413305e3a7a3036894388a356e4860353f662c7))

## 1.0.0 (2026-06-05)

### ⚠ BREAKING CHANGES

* migração para 100% deep learning PyTorch e recuperação Martingale cross-symbol em mercado único

### Funcionalidades

* **all:** backtest M15 assertivo, risco diario e guardrails macro ([339917b](https://github.com/victorh-silveira/aether-quantum-engine/commit/339917bd8c636736a0f1eea94fce634cad38f49b))
* **api:** migra autenticacao Deriv para PAT e App ID ([fbdc0e7](https://github.com/victorh-silveira/aether-quantum-engine/commit/fbdc0e76dfb15801020a7220e5614bdbf4103d08))
* **config:** altera duracao padrao de trade para M30 e reativa todos os indices ([44022c5](https://github.com/victorh-silveira/aether-quantum-engine/commit/44022c58c99d2a3a5db5c8b8ee495835614b9302))
* **config:** ativa logs detalhados LLM_IO na main ([7634162](https://github.com/victorh-silveira/aether-quantum-engine/commit/7634162eefa9ff1b997e255bcce0e0a35ba6669f))
* **config:** elevar limites de conviccao minima para 85% no settings.json ([4fe8dab](https://github.com/victorh-silveira/aether-quantum-engine/commit/4fe8dab1c37b0f8512371cfde89c4cc94a44d032))
* **config:** elevar limites de conviccao minima para 85% no settings.json ([67c3ae6](https://github.com/victorh-silveira/aether-quantum-engine/commit/67c3ae6b3f8ff8f8f8ecbc564a30e5e3f907acc6))
* **config:** exclui indice OTC_FCHI por requerer duracao de M30 no broker ([687c4f7](https://github.com/victorh-silveira/aether-quantum-engine/commit/687c4f77207e9f2655bacfd9763f8d2f79869f91))
* **config:** exclui indice OTC_GDAXI por requerer duracao minima de M30 ([f6163c0](https://github.com/victorh-silveira/aether-quantum-engine/commit/f6163c0ba1c7d2cf28bb9181db7a2fe037bbf348))
* **config:** refina prompt medallion na main ([77c0651](https://github.com/victorh-silveira/aether-quantum-engine/commit/77c0651687e89591edbe758309aca357d4d464cc))
* **engine:** amplia amostragem DL, corrige boot e refatora motor live ([5e8dc9b](https://github.com/victorh-silveira/aether-quantum-engine/commit/5e8dc9bf4ddd749d490ef3c21ac88b9ad3b16ff0))
* **engine:** Python 3.13.12, Conda deriv-api e refator DL/execucao ([2e3f2ee](https://github.com/victorh-silveira/aether-quantum-engine/commit/2e3f2ee9e309993579fc879855575b24c3647545))
* **llm:** adiciona exemplo de resposta valida no system prompt ([6aef7c8](https://github.com/victorh-silveira/aether-quantum-engine/commit/6aef7c87f47718f5d5db639b02c46d9dab9b5dac))
* **llm:** adiciona trava de entropia e melhora prompt na main ([031a7f6](https://github.com/victorh-silveira/aether-quantum-engine/commit/031a7f6d414ad3bbde36b76e03e6d115892427e9))
* **llm:** aplica melhorias de prompt e parser na main ([8456def](https://github.com/victorh-silveira/aether-quantum-engine/commit/8456defc305f7eb6322a1c9e5e4e15e9a1b69c08))
* **llm:** atualiza modelo para gemini-3.1-pro-preview ([c905210](https://github.com/victorh-silveira/aether-quantum-engine/commit/c9052108e15f21b7a781be8ee4a375622f2cebd8))
* **llm:** aumenta teto de probabilidade para 0.75 em mercados ruidosos na main ([b29e5a9](https://github.com/victorh-silveira/aether-quantum-engine/commit/b29e5a9e45b00601641ca4078ed63dd6c7de1383))
* **llm:** capacita o modelo a prever direcoes de cluster de forma independente ([cc6a594](https://github.com/victorh-silveira/aether-quantum-engine/commit/cc6a594a92a37932d0ba5ce7d58d93b9ee0d77dd))
* **llm:** confluencia macro transatlantica com RISE/FALL e somente CALL/PUT ([e11ea2e](https://github.com/victorh-silveira/aether-quantum-engine/commit/e11ea2e3665a3cd0ae17f15017fa6cc877c2396f))
* **llm:** elevar conviccao maxima em divergencia macro ([be639c0](https://github.com/victorh-silveira/aether-quantum-engine/commit/be639c009d401bd1a83f12fefb5e5c7a91291354))
* **llm:** execucao hibrida no cluster refresh e pilha quant M5 ([dc75553](https://github.com/victorh-silveira/aether-quantum-engine/commit/dc75553e67c5259611ffc8b074d7fa001b5f7fe8))
* **llm:** implementa suporte a clusters US e EU ([11aae59](https://github.com/victorh-silveira/aether-quantum-engine/commit/11aae59561e1beeadea1b4ff06119e8fb1a9a55c))
* **llm:** implementa suporte a clusters US e EU ([42fc740](https://github.com/victorh-silveira/aether-quantum-engine/commit/42fc740c958a88f44a858164a39d08d66a21e7aa))
* **llm:** implementar motor Medallion StatArb e classificador de regime HMM ([825790c](https://github.com/victorh-silveira/aether-quantum-engine/commit/825790c783ea41982e086dd5284b26cf91ab55c7))
* **llm:** injeta metricas de indice de cluster em tempo real no prompt sniper ([530bd3b](https://github.com/victorh-silveira/aether-quantum-engine/commit/530bd3b5bbda8dd3ac3574ee1f4b9af0260d30b1))
* **llm:** inversao de cluster e execucao exclusiva por macro ([c93c4f8](https://github.com/victorh-silveira/aether-quantum-engine/commit/c93c4f88ade1d81a23bf063e5eb37da5135d579b))
* **llm:** Medallion Gemini tag_change, filtros macro e backtest assertivo ([38f954a](https://github.com/victorh-silveira/aether-quantum-engine/commit/38f954aab667f877881306065a4f110caf93d1f5))
* **llm:** modo inteligencia macro pura estilo Medallion ([219284a](https://github.com/victorh-silveira/aether-quantum-engine/commit/219284adcfbda40ef98dcb58b55351e73422db93))
* **llm:** otimizar sinteticos M1 pos-win e log de inversao ([43e0469](https://github.com/victorh-silveira/aether-quantum-engine/commit/43e04698c1dfd9a8cbde676d930903015859fafb))
* **llm:** prompt com seis timeframes e propagacao por cluster ([8b3ad97](https://github.com/victorh-silveira/aether-quantum-engine/commit/8b3ad971969ddf5571378ffa1b8f82329aee00bd))
* **llm:** reduz limite de entropia para 3.0 e trava para 0.69 na main ([3479a3e](https://github.com/victorh-silveira/aether-quantum-engine/commit/3479a3ec764a421cbb297cd75a444110688d71f3))
* **llm:** refatorar motor Medallion puro com propagacao regional e StatArb ([c784300](https://github.com/victorh-silveira/aether-quantum-engine/commit/c78430062b8fc19a2d29577449968a19e03f6201))
* **llm:** remove SSMI do cluster e adiciona regras de trading no prompt ([95c67df](https://github.com/victorh-silveira/aether-quantum-engine/commit/95c67dfcb213bac1ed60a26922fd146b317723b2))
* migração para 100% deep learning PyTorch e recuperação Martingale cross-symbol em mercado único ([fdbc36e](https://github.com/victorh-silveira/aether-quantum-engine/commit/fdbc36ec546b196573055e4ae30d12d3b7a06268))
* **orchestrator:** alinhar StatArb risk_on e fallback de indices ([22dc4ed](https://github.com/victorh-silveira/aether-quantum-engine/commit/22dc4ed0e5de71e0f1eeaad0b0c3a956f31e8c75))
* **orchestrator:** backtest walk-forward e filtros de cenario lucrativo ([482f508](https://github.com/victorh-silveira/aether-quantum-engine/commit/482f508d60c69d5e8b85b5a71861bcb6725aff44))
* **orchestrator:** ciclo pos-liquidacao, refresh StatArb e inversao ([ce77260](https://github.com/victorh-silveira/aether-quantum-engine/commit/ce77260854bf8d76fe3a2b4e9a1949854b9e5c2c))
* **orchestrator:** desacopla clusters e aprimora liquidacao ([7134a14](https://github.com/victorh-silveira/aether-quantum-engine/commit/7134a14bb90ad28bdcf10872c1a5ece14ceb0094))
* **orchestrator:** folego pos-liquidacao e inversao segura ([934abf4](https://github.com/victorh-silveira/aether-quantum-engine/commit/934abf4fa2ba249daf55a6384446ae52a2d394c5))
* **orchestrator:** quarentena de inversao apos loss ([3c094f4](https://github.com/victorh-silveira/aether-quantum-engine/commit/3c094f49e6fda9c9657a62c1cb8f09d2f3ec7d8b))
* **orchestrator:** remove codigo morto e arquivo nao utilizado ([c8f6e60](https://github.com/victorh-silveira/aether-quantum-engine/commit/c8f6e608742285e082c0830fb30adad6641000a2))
* **orchestrator:** resolve rate limit de proposta e otimiza clusters ([779da44](https://github.com/victorh-silveira/aether-quantum-engine/commit/779da44c1f720cb95f4a766fdc7d0fb90d3194d3))
* **repo:** commit inicial do projeto ([0d35232](https://github.com/victorh-silveira/aether-quantum-engine/commit/0d3523284600179317b157321d74563287d2cbf8))
* **repo:** remove arquivo de changelog ([e1491a0](https://github.com/victorh-silveira/aether-quantum-engine/commit/e1491a03a6a04ee203a066c9d3dc92f7268d5906))
* **risk-statarb:** otimizacao do medallion, remocao de limites de kelly/recuperacao e ativacao de clusters transatlanticos ([beb0e63](https://github.com/victorh-silveira/aether-quantum-engine/commit/beb0e63a3daf0acfeaca1e23a731a1f79b1bfea5))
* **risk:** adiciona controle de perdas consecutivas e escalonamento de cooldown ([fe08dac](https://github.com/victorh-silveira/aether-quantum-engine/commit/fe08dac239e4b2df9eaa3f3ef9062a8bd7daa8f5))
* **risk:** implementa logica cirurgica do single strike e volatilidade dinamica HMM ([ad5ee46](https://github.com/victorh-silveira/aether-quantum-engine/commit/ad5ee462bfcee6b9bdd534422fe5c4f8d650a288))
* **risk:** otimiza modelo de entrada unica para stop win em 30m e ajusta prompts e inversao ([26b972d](https://github.com/victorh-silveira/aether-quantum-engine/commit/26b972da0be89e62007c7faacf2a9f7b2cc5e08c))
* **risk:** otimizar parametros do Medallion para recuperacao, frequencia e escopo de cluster ([be9a7d0](https://github.com/victorh-silveira/aether-quantum-engine/commit/be9a7d09f91139454b3d2105cbb9e95844998be2))
* **risk:** sessao OTC, stake tier e refresh LLM periodico ([5adc092](https://github.com/victorh-silveira/aether-quantum-engine/commit/5adc09210e32d8bfbe623f62633e56283b53054d))
* **scripts:** adiciona pylint para deteccao nativa de codigo duplicado ([9b65171](https://github.com/victorh-silveira/aether-quantum-engine/commit/9b6517159362cabd44058234f6da889bcb27c5ce))

### Correcoes de Bug

* **config:** corrige erro de decode no windows e ativa modo live ([189777f](https://github.com/victorh-silveira/aether-quantum-engine/commit/189777f46900adf82fe416e1ab145aae79c2b0f3))
* **dl:** corrige path do mock no teste de coleta de deep learning ([81caf13](https://github.com/victorh-silveira/aether-quantum-engine/commit/81caf13c203328d3d8abd6168cbc8289b5e0d35e))
* **engine:** alinhar clusters ao macro e corrigir liquidacao e persistencia ([27bb693](https://github.com/victorh-silveira/aether-quantum-engine/commit/27bb693b177e30004d15aa011ae5d1296e763903))
* **llm:** aceitar conviccao LLM em divergencia sem piso por tag ([6e5ac93](https://github.com/victorh-silveira/aether-quantum-engine/commit/6e5ac93bb48b1b95de00636040f72f10af598a02))
* **llm:** desativa thinking e corrige truncamento MAX_TOKENS ([f1fadbd](https://github.com/victorh-silveira/aether-quantum-engine/commit/f1fadbd6095016cd96fdd41eababd32c10b1ecb4))
* **llm:** ignorar ruido US em divergencia e pular cluster flat ([2822d57](https://github.com/victorh-silveira/aether-quantum-engine/commit/2822d57111c3c97440132a4ccf51989aee45cd0b))
* **llm:** inverter lado LLM e contratos M1 nos sinteticos ([96057f8](https://github.com/victorh-silveira/aether-quantum-engine/commit/96057f8540ec05d40acb1939074d8876aaf23044))
* **llm:** isolamento do parser de decisoes e garantia de cobertura ([789b312](https://github.com/victorh-silveira/aether-quantum-engine/commit/789b3123b36552b2d4a5246ed331d16ecefb702a))
* **llm:** liberar trades apos loss com cluster refresh risk_off ([3694de3](https://github.com/victorh-silveira/aether-quantum-engine/commit/3694de36cd0a8c884107115af79799f1f66147cc))
* **llm:** manter Gemini ativo em indefinido e sem pausa pos-loss ([61a47f5](https://github.com/victorh-silveira/aether-quantum-engine/commit/61a47f588eafadd22c57c3ba6062cb988a03d5c0))
* **llm:** preservar direcao Gemini em tags de divergencia ([4fcb96a](https://github.com/victorh-silveira/aether-quantum-engine/commit/4fcb96a6e6acd44492f5500f3bb244454de74095))
* **llm:** propagacao por indice, parser US/EU compacto e log LLM_IO off ([2d471be](https://github.com/victorh-silveira/aether-quantum-engine/commit/2d471be850bd19181eba50ab6b2c17e4eaeced63))
* **llm:** reduz timeouts Gemini com flash, menos retries e fallback ([fd70232](https://github.com/victorh-silveira/aether-quantum-engine/commit/fd702323fa1f7272fe7d0e4c363c307572837f95))
* **llm:** reforcar vetos e penalidades em divergencia macro ([6cb6476](https://github.com/victorh-silveira/aether-quantum-engine/commit/6cb6476dcd3be7ba81a0c59ac34b6b43a342300c))
* **llm:** remove fallback de indices e exige CALL/PUT somente da LLM ([5b7eb43](https://github.com/victorh-silveira/aether-quantum-engine/commit/5b7eb431ebd0d2912507fc549130d94a55047255))
* **llm:** saida JSON obrigatoria CALL/PUT e fallback flash-lite ([04ee185](https://github.com/victorh-silveira/aether-quantum-engine/commit/04ee18583ac8a7b335214b6f3d96fe615ce70564))
* **llm:** trocar vetos HMM por inteligencia e liberar divergencia LLM ([6e8cf2f](https://github.com/victorh-silveira/aether-quantum-engine/commit/6e8cf2fe20075c2bc91ed4aa0fe94644d5b9bcae))
* **orchestrator:** move set_trading(False) para dentro do bloco de stop win na main ([9ff1af2](https://github.com/victorh-silveira/aether-quantum-engine/commit/9ff1af2ed29cb81c21258309f35b83fbaac1b75d))
* **risk:** martingale dobra ultima entrada e setup WSL ([e04e3af](https://github.com/victorh-silveira/aether-quantum-engine/commit/e04e3afe546e83dc0978cf0ee8d98d94ac126233))
* **risk:** suavizar Kelly e desativar stop-win agressivo ([77fa602](https://github.com/victorh-silveira/aether-quantum-engine/commit/77fa60275619dca74f069bae222d235684b782fa))
* **scripts:** ajustar atalho do run.py nos scripts de launch ([fd5b189](https://github.com/victorh-silveira/aether-quantum-engine/commit/fd5b18909e8f899e5c7a5619b062f765c33419a7))
* **scripts:** atualiza script de ping para gemini-3.1-pro-preview ([b08ce22](https://github.com/victorh-silveira/aether-quantum-engine/commit/b08ce22c413fb11a8e82af2346ca41c6427ced06))
* **scripts:** resolve python do venv do projeto para quality gates ([d150c76](https://github.com/victorh-silveira/aether-quantum-engine/commit/d150c761cdb19f14c5bd9aaa954f1c6e7bbcaa0c))
* **test:** adiciona cobertura para o load_dotenv no auth_manager ([0e5af0d](https://github.com/victorh-silveira/aether-quantum-engine/commit/0e5af0d3270886d770f6a77691b662bf331fc656))
* **test:** adiciona teste para early stopping e corrige resolve_conda_python ([64d0ab0](https://github.com/victorh-silveira/aether-quantum-engine/commit/64d0ab03e4f3e5cc9ede9eac000facb313c496fd))
* **tools:** hooks WSL, refator lint e docstrings Medallion ([a716f0c](https://github.com/victorh-silveira/aether-quantum-engine/commit/a716f0cbd44a3a01508aa2b4d02a00871704a4df))

### Refatoracoes Tecnicas

* **backtest:** remove completamente o sistema de backtest offline e documentacao ([28127ef](https://github.com/victorh-silveira/aether-quantum-engine/commit/28127ef313ed92c1adda41323734ae9025418bd8))
* **config:** adiciona filtro m5 e aumenta thresholds ([9f022dc](https://github.com/victorh-silveira/aether-quantum-engine/commit/9f022dc620b66386175f5177de882a6525efe324))
* **config:** atualiza versao do Python para 3.14.5 ([d62c901](https://github.com/victorh-silveira/aether-quantum-engine/commit/d62c9010d3be2d8e5f36252c2b59c5f20243dec6))
* **config:** expande as verificacoes de qualidade para todos os arquivos python ([325795c](https://github.com/victorh-silveira/aether-quantum-engine/commit/325795c02cefd15a8c22b95aced52f14aad8a3aa))
* **config:** remove nomes ignorados do vulture ([9845048](https://github.com/victorh-silveira/aether-quantum-engine/commit/984504806076974627eefdacd556d4d039f698b7))
* **llm:** refina diretrizes quant do system_prompt para reversao de zscore extremo ([bc753d5](https://github.com/victorh-silveira/aether-quantum-engine/commit/bc753d5f4882dbcc426e6fc739e51a732eea7530))
* **llm:** remove chaves de config inativas e adiciona verificacoes de websocket ([bbe1684](https://github.com/victorh-silveira/aether-quantum-engine/commit/bbe1684c8792c3f598bb398e8af0fd560fc2c7bd))
* **llm:** remove fallback de execucao e aprimora prompt do gemini ([6a15914](https://github.com/victorh-silveira/aether-quantum-engine/commit/6a15914b80083db29b87ead07640fcfad74fdca6))
* **repo:** layout app, docs e linters sem infra K8s ([b2a677e](https://github.com/victorh-silveira/aether-quantum-engine/commit/b2a677e34d73d4490a4b33c1de9bf986e7e6c41d))
* **test:** atualiza suite de testes unitarios para simbolos ativos ([387c175](https://github.com/victorh-silveira/aether-quantum-engine/commit/387c175ec20e4175de6dedf1dcdab9c1f89c5737))

### Documentacao

* **deriv:** adiciona documentacao do algoritmo CSPRNG da Deriv e corrige referencias a simbolos legados ([8825a9f](https://github.com/victorh-silveira/aether-quantum-engine/commit/8825a9f855e5dda895591ee90481d8ff4fbfaa5a))
* indicar branch sinteticos no README main ([165e263](https://github.com/victorh-silveira/aether-quantum-engine/commit/165e263aab4f535fe65a82fae329ba7e7d260fd4))
* **release:** remover secao Unreleased do CHANGELOG ([6c5b23f](https://github.com/victorh-silveira/aether-quantum-engine/commit/6c5b23f1ba7934deb8b143fba41199c7f65c674a))
* remove referencias a branch de indices sinteticos ([6aa9200](https://github.com/victorh-silveira/aether-quantum-engine/commit/6aa9200065a1d9e2db922e9dc9fa1d9de3afb6e9))
* **repo:** atualiza README.md para modelo medallion na main ([8c56109](https://github.com/victorh-silveira/aether-quantum-engine/commit/8c56109b74cbec6506b65f1e819353e0b8dd529e))

## 1.0.0 (2026-06-02)

### ⚠ BREAKING CHANGES

* migração para 100% deep learning PyTorch e recuperação Martingale cross-symbol em mercado único

### Funcionalidades

* **all:** backtest M15 assertivo, risco diario e guardrails macro ([7f4c096](https://github.com/victorh-silveira/aether-quantum-engine/commit/7f4c096e4ffdf25d606aae15315e8d8c68493dc2))
* **config:** altera duracao padrao de trade para M30 e reativa todos os indices ([521a7e6](https://github.com/victorh-silveira/aether-quantum-engine/commit/521a7e6af38fa42c9e8a4fd315a7d3890fb80c56))
* **config:** ativa logs detalhados LLM_IO na main ([be1b1e9](https://github.com/victorh-silveira/aether-quantum-engine/commit/be1b1e9993d645be29399df9f713e3860834fd84))
* **config:** elevar limites de conviccao minima para 85% no settings.json ([8a69dcb](https://github.com/victorh-silveira/aether-quantum-engine/commit/8a69dcb6e349f8d3d41ce2f1e66f2b6dcfec9a16))
* **config:** elevar limites de conviccao minima para 85% no settings.json ([907f93c](https://github.com/victorh-silveira/aether-quantum-engine/commit/907f93ce5c261e7cde384573c1f6122f15970171))
* **config:** exclui indice OTC_FCHI por requerer duracao de M30 no broker ([5a66d58](https://github.com/victorh-silveira/aether-quantum-engine/commit/5a66d5894bd14a21dcbae2417d459d954496e292))
* **config:** exclui indice OTC_GDAXI por requerer duracao minima de M30 ([024dd0b](https://github.com/victorh-silveira/aether-quantum-engine/commit/024dd0b340815449d3a0aa904dd1806e2e4f0aa7))
* **config:** refina prompt medallion na main ([86756a9](https://github.com/victorh-silveira/aether-quantum-engine/commit/86756a9764b3b5d465536ade1770f01e1dfda651))
* **llm:** adiciona exemplo de resposta valida no system prompt ([0f061a8](https://github.com/victorh-silveira/aether-quantum-engine/commit/0f061a80265222f111fc09cac50449a9c628e777))
* **llm:** adiciona trava de entropia e melhora prompt na main ([9318468](https://github.com/victorh-silveira/aether-quantum-engine/commit/9318468f24737264f442157f386fee4b903ff182))
* **llm:** aplica melhorias de prompt e parser na main ([7d295aa](https://github.com/victorh-silveira/aether-quantum-engine/commit/7d295aaa3f3d235160b9e869dbfdfce4858f4902))
* **llm:** atualiza modelo para gemini-3.1-pro-preview ([04da19c](https://github.com/victorh-silveira/aether-quantum-engine/commit/04da19c8241dfc22695ab879aa2462af1099f3e6))
* **llm:** aumenta teto de probabilidade para 0.75 em mercados ruidosos na main ([c9aaaa5](https://github.com/victorh-silveira/aether-quantum-engine/commit/c9aaaa5f47c5228bbc203ad2dd37704875056c8e))
* **llm:** confluencia macro transatlantica com RISE/FALL e somente CALL/PUT ([8027862](https://github.com/victorh-silveira/aether-quantum-engine/commit/80278627b84501ad8b5cec896f10bff84e0004bb))
* **llm:** elevar conviccao maxima em divergencia macro ([6598759](https://github.com/victorh-silveira/aether-quantum-engine/commit/6598759cfba6413c471c7b4a1e3170c682d2a48c))
* **llm:** empower model to predict cluster directions independently ([2c5dc3b](https://github.com/victorh-silveira/aether-quantum-engine/commit/2c5dc3ba9bf6a3a6d6a9cfb821a67835349a0538))
* **llm:** execucao hibrida no cluster refresh e pilha quant M5 ([35491bd](https://github.com/victorh-silveira/aether-quantum-engine/commit/35491bdbb67525d1512a053ec01ca1dc466928d6))
* **llm:** implementa suporte a clusters US e EU ([5931f64](https://github.com/victorh-silveira/aether-quantum-engine/commit/5931f6415bd2da021c0bd849c9da5d32a4d88cc4))
* **llm:** implementa suporte a clusters US e EU ([565f46a](https://github.com/victorh-silveira/aether-quantum-engine/commit/565f46a4642b13faedd9d662a6109ecc61263998))
* **llm:** implementar motor Medallion StatArb e classificador de regime HMM ([d1c3586](https://github.com/victorh-silveira/aether-quantum-engine/commit/d1c35864069cfa910fdfc81a281ef9f8cbd50ee6))
* **llm:** inject realtime cluster index metrics into sniper prompt ([683e6e2](https://github.com/victorh-silveira/aether-quantum-engine/commit/683e6e2a41d0bfc7567108c181acc9cc41598465))
* **llm:** inversao de cluster e execucao exclusiva por macro ([d15b451](https://github.com/victorh-silveira/aether-quantum-engine/commit/d15b451b41e090985f852409b86377203d1284e0))
* **llm:** Medallion Gemini tag_change, filtros macro e backtest assertivo ([8ccccf2](https://github.com/victorh-silveira/aether-quantum-engine/commit/8ccccf246e3fb8ca0137656d8cfcae2e0fcd1506))
* **llm:** modo inteligencia macro pura estilo Medallion ([7a1b195](https://github.com/victorh-silveira/aether-quantum-engine/commit/7a1b1959b914878ebc715bf3c2ea47d4eeeba8dd))
* **llm:** otimizar sinteticos M1 pos-win e log de inversao ([ce96e88](https://github.com/victorh-silveira/aether-quantum-engine/commit/ce96e88f3c2b844f6fbdb1f87d1161190005bf73))
* **llm:** prompt com seis timeframes e propagacao por cluster ([e4b4f42](https://github.com/victorh-silveira/aether-quantum-engine/commit/e4b4f4285ab98a9fd2b73763f2f12d2c0f513323))
* **llm:** reduz limite de entropia para 3.0 e trava para 0.69 na main ([91f6b34](https://github.com/victorh-silveira/aether-quantum-engine/commit/91f6b3455e6ae357d8ffeacc6a348bf9a604bd3a))
* **llm:** refatorar motor Medallion puro com propagacao regional e StatArb ([efd4fba](https://github.com/victorh-silveira/aether-quantum-engine/commit/efd4fba1f777648db318326f17acc5f12a6aecfe))
* **llm:** remove SSMI do cluster e adiciona regras de trading no prompt ([e6ce56a](https://github.com/victorh-silveira/aether-quantum-engine/commit/e6ce56af7a2257005be8183f7a11861c3a9831ba))
* migração para 100% deep learning PyTorch e recuperação Martingale cross-symbol em mercado único ([9e522e8](https://github.com/victorh-silveira/aether-quantum-engine/commit/9e522e869724d88f9eed4d1c62e24fa8a8e7cbab))
* **orchestrator:** alinhar StatArb risk_on e fallback de indices ([9bc1bc8](https://github.com/victorh-silveira/aether-quantum-engine/commit/9bc1bc808126f6afa4178b8b885a407c3583d5ed))
* **orchestrator:** backtest walk-forward e filtros de cenario lucrativo ([446dd01](https://github.com/victorh-silveira/aether-quantum-engine/commit/446dd015d2cb4f6500f46a6c31ec1996e3868dc5))
* **orchestrator:** ciclo pos-liquidacao, refresh StatArb e inversao ([e21d5f5](https://github.com/victorh-silveira/aether-quantum-engine/commit/e21d5f5c1743066217d29ca6408dedee7c6cb279))
* **orchestrator:** desacopla clusters e aprimora liquidacao ([84b6fd9](https://github.com/victorh-silveira/aether-quantum-engine/commit/84b6fd94bc9e02c0074a965ab5939ecdb605e96c))
* **orchestrator:** folego pos-liquidacao e inversao segura ([ccc2bde](https://github.com/victorh-silveira/aether-quantum-engine/commit/ccc2bde1ef07f57bf57eb1478f267a50ebc7cc13))
* **orchestrator:** quarentena de inversao apos loss ([59d898a](https://github.com/victorh-silveira/aether-quantum-engine/commit/59d898a1bb5dbb1aa928f66af189c0ec1956d7ea))
* **orchestrator:** remove codigo morto e arquivo nao utilizado ([3e76c58](https://github.com/victorh-silveira/aether-quantum-engine/commit/3e76c58a7983c2d39b5175f7c59024aa16810a96))
* **orchestrator:** resolve rate limit de proposta e otimiza clusters ([4188316](https://github.com/victorh-silveira/aether-quantum-engine/commit/4188316a976c6521c22429829767c4cac9435672))
* **repo:** commit inicial do projeto ([0f10d2f](https://github.com/victorh-silveira/aether-quantum-engine/commit/0f10d2f9815bea3115552c056c32af6bef4daf7c))
* **repo:** remove arquivo de changelog ([0f398d5](https://github.com/victorh-silveira/aether-quantum-engine/commit/0f398d51d03f8faad1713fb85e0974f858408e05))
* **risk-statarb:** otimizacao do medallion, remocao de limites de kelly/recuperacao e ativacao de clusters transatlanticos ([d46584a](https://github.com/victorh-silveira/aether-quantum-engine/commit/d46584a5dccdf25b574045bd699e1ffef9b6003b))
* **risk:** adiciona controle de perdas consecutivas e escalonamento de cooldown ([bb22d19](https://github.com/victorh-silveira/aether-quantum-engine/commit/bb22d198e0bf0695406afd43525c1a3794d27f3d))
* **risk:** implementa logica cirurgica do single strike e volatilidade dinamica HMM ([1ebb07a](https://github.com/victorh-silveira/aether-quantum-engine/commit/1ebb07a0f63abb9cf324f8ff2639849fcef73ca7))
* **risk:** otimiza modelo de entrada unica para stop win em 30m e ajusta prompts e inversao ([017bd93](https://github.com/victorh-silveira/aether-quantum-engine/commit/017bd9321cbc5142204d486ef93ef635d50e8742))
* **risk:** otimizar parametros do Medallion para recuperacao, frequencia e escopo de cluster ([13231df](https://github.com/victorh-silveira/aether-quantum-engine/commit/13231dff60b081bb6378275d3d64ac82d2233441))
* **risk:** sessao OTC, stake tier e refresh LLM periodico ([d85f9bb](https://github.com/victorh-silveira/aether-quantum-engine/commit/d85f9bbac6485905253d7679c34aec4d6a238157))
* **scripts:** add pylint for native duplicate code detection ([fbce041](https://github.com/victorh-silveira/aether-quantum-engine/commit/fbce041062e4e6a9ee0bb8206fd8cb31c358c99a))

### Correcoes de Bug

* **config:** corrige erro de decode no windows e ativa modo live ([4c9a659](https://github.com/victorh-silveira/aether-quantum-engine/commit/4c9a6590827389a29b46fa255f4b0e3d13f48190))
* **engine:** alinhar clusters ao macro e corrigir liquidacao e persistencia ([d57e91c](https://github.com/victorh-silveira/aether-quantum-engine/commit/d57e91c9da99a75126c12a11f2657f1daae1af23))
* **llm:** aceitar conviccao LLM em divergencia sem piso por tag ([0f9baaf](https://github.com/victorh-silveira/aether-quantum-engine/commit/0f9baaf650526b48c2062ca6e907306c0de3d886))
* **llm:** desativa thinking e corrige truncamento MAX_TOKENS ([ee936a8](https://github.com/victorh-silveira/aether-quantum-engine/commit/ee936a8b83a56d68e121d023dbdf551deea495a6))
* **llm:** ignorar ruido US em divergencia e pular cluster flat ([9e2ef21](https://github.com/victorh-silveira/aether-quantum-engine/commit/9e2ef215efca60f4cdbd9bf1d33726e0045f60a7))
* **llm:** inverter lado LLM e contratos M1 nos sinteticos ([7d917ed](https://github.com/victorh-silveira/aether-quantum-engine/commit/7d917ed38832e32dbecc4a8d36d1edf6a2e94487))
* **llm:** saida JSON obrigatoria CALL/PUT e fallback flash-lite ([775c79c](https://github.com/victorh-silveira/aether-quantum-engine/commit/775c79cd4a2c052ab03da91aa2c773d2e89e179f))
* **llm:** trocar vetos HMM por inteligencia e liberar divergencia LLM ([f3e4a61](https://github.com/victorh-silveira/aether-quantum-engine/commit/f3e4a6119837b96f8c8279e05ec522862dd54b7c))
* **orchestrator:** move set_trading(False) para dentro do bloco de stop win na main ([95aba1d](https://github.com/victorh-silveira/aether-quantum-engine/commit/95aba1dfa576b1c6df814ae9df8e216182676120))
* **risk:** suavizar Kelly e desativar stop-win agressivo ([7c40611](https://github.com/victorh-silveira/aether-quantum-engine/commit/7c40611337ad8c0e934072e221b2dafe31879067))
* **scripts:** ajustar atalho do run.py nos scripts de launch ([2e6acb5](https://github.com/victorh-silveira/aether-quantum-engine/commit/2e6acb5885e7a015c454e8db69bed1675cd4de96))
* **scripts:** resolve project venv python for quality gates ([77804b3](https://github.com/victorh-silveira/aether-quantum-engine/commit/77804b327116eac1ea36f78836837ea0b87496ec))
* **scripts:** update ping script for gemini-3.1-pro-preview ([38681a3](https://github.com/victorh-silveira/aether-quantum-engine/commit/38681a38bad3e49b5029d3b5fe72568596ccaff2))
* **tools:** hooks WSL, refator lint e docstrings Medallion ([f949620](https://github.com/victorh-silveira/aether-quantum-engine/commit/f9496208cf5e457148ffffdb112ba2d404b9368c))

### Refatoracoes Tecnicas

* **config:** adiciona filtro m5 e aumenta thresholds ([91fe2de](https://github.com/victorh-silveira/aether-quantum-engine/commit/91fe2de6b0aca4af1693b59b83b3b64fb3c44d72))
* **config:** expand quality checks to scan all python files ([2acc9c3](https://github.com/victorh-silveira/aether-quantum-engine/commit/2acc9c37b0dbddb140e229cc426617c6e6f4cc57))
* **config:** remove vulture ignored names ([a502561](https://github.com/victorh-silveira/aether-quantum-engine/commit/a50256105543ccfcfb1a141b25b54159dd5156cc))
* **config:** update Python version to 3.14.5 ([0558196](https://github.com/victorh-silveira/aether-quantum-engine/commit/05581962b8ee846de1fa4de7b371a399639845db))
* **llm:** refina diretrizes quant do system_prompt para reversao de zscore extremo ([9c15dbd](https://github.com/victorh-silveira/aether-quantum-engine/commit/9c15dbd1f7e556c6669a8b8b282f143afb41a39e))
* **llm:** remove fallback de execucao e aprimora prompt do gemini ([4261607](https://github.com/victorh-silveira/aether-quantum-engine/commit/42616073da377a8c88af94301cbc81ff6ade1a77))
* **llm:** remove inactive config keys and add websocket checks ([9eb8f49](https://github.com/victorh-silveira/aether-quantum-engine/commit/9eb8f49f50d09e7944d1269a522bded5cc1a9959))
* **repo:** layout app, docs e linters sem infra K8s ([af35c6d](https://github.com/victorh-silveira/aether-quantum-engine/commit/af35c6d1d7ea12faa74923e153502dab99349fdc))
* **test:** atualiza suite de testes unitarios para simbolos ativos ([a3d9461](https://github.com/victorh-silveira/aether-quantum-engine/commit/a3d9461afed6db8c0f937ffe3145f7c92e227f22))

### Documentacao

* indicar branch sinteticos no README main ([add25b9](https://github.com/victorh-silveira/aether-quantum-engine/commit/add25b99b84430cd53f4039704a8a168dc2a0fe8))
* **release:** remover secao Unreleased do CHANGELOG ([cb635c7](https://github.com/victorh-silveira/aether-quantum-engine/commit/cb635c75c09cc072ee61173657c985e7139c9b75))
* remove referencias a branch de indices sinteticos ([ceb2fc0](https://github.com/victorh-silveira/aether-quantum-engine/commit/ceb2fc0152f09ec59fca31541ea14b07ea681779))
* **repo:** atualiza README.md para modelo medallion na main ([aafe3d7](https://github.com/victorh-silveira/aether-quantum-engine/commit/aafe3d7fd33d4afa8d7cdfb0d3b8a0ea1d64b5b2))
