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
