## [1.30.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.29.0...v1.30.0) (2026-05-15)

### Funcionalidades

* **orchestrator:** expand cluster and refine risk thresholds ([34877b1](https://github.com/victorh-silveira/aether-quantum-engine/commit/34877b14c27c887d3e57efd89bd8561cbf171c21))

## [1.29.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.28.0...v1.29.0) (2026-05-15)

### Funcionalidades

* **orchestrator:** migracao para indices europeus e liquidacao dinamica ([d274679](https://github.com/victorh-silveira/aether-quantum-engine/commit/d274679f90051195de61db1471fbe11c6ff798ad))

## [1.28.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.27.0...v1.28.0) (2026-05-13)

### Funcionalidades

* **orchestrator:** extingue o modo IDLE com fail-safes absolutos ([2019da2](https://github.com/victorh-silveira/aether-quantum-engine/commit/2019da20a2bf687811c22e41f1bf44f4273bc0b5))

## [1.27.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.26.0...v1.27.0) (2026-05-13)

### Funcionalidades

* **orchestrator:** habilita modo de execução absoluta e proíbe SKIP ([21e8e36](https://github.com/victorh-silveira/aether-quantum-engine/commit/21e8e361b537d27ef3b504b0eded10d79f7a4a69))

## [1.26.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.25.0...v1.26.0) (2026-05-13)

### Funcionalidades

* **orchestrator:** restaura agressividade do kelly e reduz gates de convicção ([ab3574a](https://github.com/victorh-silveira/aether-quantum-engine/commit/ab3574ae7606975c2bc84d920996154872eecdec))

## [1.25.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.24.0...v1.25.0) (2026-05-13)

### Funcionalidades

* **orchestrator:** endurece filtros de convicção e proteção contra ruído ([1b3e0b9](https://github.com/victorh-silveira/aether-quantum-engine/commit/1b3e0b90240018b2d901757b2eeeaa15a84191c3))

## [1.24.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.23.0...v1.24.0) (2026-05-13)

### Funcionalidades

* **orchestrator:** otimiza sincronização de estado e monitoramento LLM ([034fbba](https://github.com/victorh-silveira/aether-quantum-engine/commit/034fbba2e8944520947f561e6958ca40f98fb7f2))

### Correcoes de Bug

* **orchestrator:** corrige import do monitor e moderniza lógica de caminhos ([8a64fa5](https://github.com/victorh-silveira/aether-quantum-engine/commit/8a64fa52eb5c9d54772d555cc3617a0eced58ccb))

## [1.24.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.23.0...v1.24.0) (2026-05-13)

### Funcionalidades

* **orchestrator:** optimize state sync and llm monitoring ([19a2640](https://github.com/victorh-silveira/aether-quantum-engine/commit/19a26406ac09a585c274b4a5cd865fd9899d0bd6))

## [1.23.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.22.3...v1.23.0) (2026-05-13)

### Funcionalidades

* **config:** troca para Australian 200 (OTC_AS51) e risco agressivo ([a0ce34c](https://github.com/victorh-silveira/aether-quantum-engine/commit/a0ce34c23c223006667fed26fc585aa2e4aaeea8))

## [1.22.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.22.2...v1.22.3) (2026-05-13)

### Correcoes de Bug

* **risk:** ajusta payout_estimate para Kelly positivo ([63d48b9](https://github.com/victorh-silveira/aether-quantum-engine/commit/63d48b976b6bfec86b96fb744f7ecd09b1ae9f25))

## [1.22.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.22.1...v1.22.2) (2026-05-13)

### Correcoes de Bug

* **risk:** altera duracao para 15m na main ([1273186](https://github.com/victorh-silveira/aether-quantum-engine/commit/1273186bc37ffe8575268a5f41904d597ee6c551))

## [1.22.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.22.0...v1.22.1) (2026-05-13)

### Correcoes de Bug

* **risk:** reverte para RISE_FALL na main ([dd52c76](https://github.com/victorh-silveira/aether-quantum-engine/commit/dd52c76c38a5c7840ca5960e3a3ef3ed46674fef))

## [1.22.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.21.0...v1.22.0) (2026-05-13)

### Funcionalidades

* **engine:** configura mercado real EURUSD ([54993b7](https://github.com/victorh-silveira/aether-quantum-engine/commit/54993b7642912d7474204333f387fe7318c8fee8))
* **engine:** integra multiplicadores Deriv com x800 e meta de 3% ([66f54dd](https://github.com/victorh-silveira/aether-quantum-engine/commit/66f54dd64da13631e1b18b3e49e659f23e2c94ae))
* **engine:** sincroniza main com melhorias de multiplicadores ([c20abdf](https://github.com/victorh-silveira/aether-quantum-engine/commit/c20abdf7427a499378de15bc2a4bf784f6f3e08f))
* **llm:** implementa politica sempre operar ([8fcc171](https://github.com/victorh-silveira/aether-quantum-engine/commit/8fcc171f27c9a5c7645228f96fe1162e6b28ac0f))

### Correcoes de Bug

* **engine:** limita take_profit em ([933e908](https://github.com/victorh-silveira/aether-quantum-engine/commit/933e908742f9343ec8d0ab86c3c5c1a7ac96bef9))
* **engine:** permite duracao 'MULT' no settlement_utils ([e68f594](https://github.com/victorh-silveira/aether-quantum-engine/commit/e68f594a11ca3bdebd6e21efffc40498ee77c47f))
* **engine:** remove remaining int() conversions for multiplier duration ([4f70dc2](https://github.com/victorh-silveira/aether-quantum-engine/commit/4f70dc2264ad9b2dd6aff513d6d031e601c1fcfd))
* **engine:** usa 'symbol' em vez de 'underlying_symbol' para multiplicadores ([4e41dfe](https://github.com/victorh-silveira/aether-quantum-engine/commit/4e41dfe0b6e8c4e6de78f72b8dc05bef849f113b))
* **risk:** ajusta stake e limite de TP para multiplicadores ([3405647](https://github.com/victorh-silveira/aether-quantum-engine/commit/3405647ea759a07dfc6dc2e8cdba3b74f0009143))

## [1.21.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.20.1...v1.21.0) (2026-05-13)

### Funcionalidades

* **engine:** setup EURUSD real market configuration ([831cab2](https://github.com/victorh-silveira/aether-quantum-engine/commit/831cab227cf08f6858dc62b1a3766102a6cd23e4))

## [1.20.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.20.0...v1.20.1) (2026-05-13)

## [1.20.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.19.0...v1.20.0) (2026-05-13)

### Funcionalidades

* **risk:** implementa Anti-Manipulation Guard para Deriv ([4641a9d](https://github.com/victorh-silveira/aether-quantum-engine/commit/4641a9dc1279e1148fbe23e81c9a5d5b697b0cb0))

## [1.19.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.18.0...v1.19.0) (2026-05-13)

### Funcionalidades

* **config:** reduz piso de execucao para eliminar IDLE excessivo ([c1c8355](https://github.com/victorh-silveira/aether-quantum-engine/commit/c1c8355673bf40cd023a0c8d2216dc5e5dcd738a))

## [1.18.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.17.0...v1.18.0) (2026-05-13)

### Funcionalidades

* **llm:** otimiza thresholds para volatilidade M1 ([7e772da](https://github.com/victorh-silveira/aether-quantum-engine/commit/7e772da485a495d0d66706d4a4aaa6204a58f49a))

## [1.17.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.16.1...v1.17.0) (2026-05-13)

### Funcionalidades

* **config:** centraliza configuracao e limpa hardcoded ([c06b082](https://github.com/victorh-silveira/aether-quantum-engine/commit/c06b08259082347e6c2ce1766609b37b0fad4942))

## [1.16.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.16.0...v1.16.1) (2026-05-13)

### Correcoes de Bug

* **llm:** aplica filtro de piso e restaura cobertura ([313525a](https://github.com/victorh-silveira/aether-quantum-engine/commit/313525a4d3e198cf935be4aa49703d61252353e2))

## [1.16.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.15.0...v1.16.0) (2026-05-13)

### Funcionalidades

* **llm:** eleva piso de execucao para 0.58 ([7a29c60](https://github.com/victorh-silveira/aether-quantum-engine/commit/7a29c60e7f42a9097c7d4156c884b2e0d0454220))

## [1.15.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.14.0...v1.15.0) (2026-05-13)

### Funcionalidades

* **llm:** recalibra thresholds para 0.75 ([c981c7a](https://github.com/victorh-silveira/aether-quantum-engine/commit/c981c7ad807a0a37a14e3657fd8219fa288853d8))

## [1.14.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.13.0...v1.14.0) (2026-05-13)

### Funcionalidades

* **llm:** remove skips e forca decisao binaria ([a7ac079](https://github.com/victorh-silveira/aether-quantum-engine/commit/a7ac079d20c57f6d3e1e416d8ee573aa08781e93))

## [1.13.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.12.0...v1.13.0) (2026-05-13)

### Funcionalidades

* **risk:** elimina estados idle forcando stake minima ([da2fd05](https://github.com/victorh-silveira/aether-quantum-engine/commit/da2fd050514d7a8bd82d7c4e819fa5a333610a18))

## [1.12.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.11.0...v1.12.0) (2026-05-13)

### Funcionalidades

* **llm:** garante inversao e telemetria no log ([1ab1701](https://github.com/victorh-silveira/aether-quantum-engine/commit/1ab17017d91e7c82e6c852faa6192ec0825985e5))

## [1.11.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.10.1...v1.11.0) (2026-05-12)

### Funcionalidades

* **llm:** identifica trades invertidos no log ([130ab32](https://github.com/victorh-silveira/aether-quantum-engine/commit/130ab32bce9aa46d79f0e6afd10167dd88dffcaf))

## [1.10.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.10.0...v1.10.1) (2026-05-12)

### Correcoes de Bug

* **llm:** corrige propagacao de thresholds ([87918f6](https://github.com/victorh-silveira/aether-quantum-engine/commit/87918f6f155e30a244deeb02b0e0e2767cda613a))

## [1.10.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.9.0...v1.10.0) (2026-05-12)

### Funcionalidades

* **llm:** implementa inversao agressiva ([3769543](https://github.com/victorh-silveira/aether-quantum-engine/commit/3769543aa88362cd6ae8dccdd5e3b8650db700ae))

## [1.9.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.8.0...v1.9.0) (2026-05-12)

### Funcionalidades

* **llm:** eliminacao total de SKIPS por Random Walk ([0ebb8fe](https://github.com/victorh-silveira/aether-quantum-engine/commit/0ebb8fed640195a35a313c97511f88381df66acc))

## [1.8.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.7.1...v1.8.0) (2026-05-12)

### Funcionalidades

* **llm:** reduz range de rWalk para destravar trades ([55d5d52](https://github.com/victorh-silveira/aether-quantum-engine/commit/55d5d52f19c2bbd7a68ab84e19e570e4a8881983))

## [1.7.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.7.0...v1.7.1) (2026-05-12)

### Correcoes de Bug

* **llm:** reequilibra filtros para reduzir perdas ([3c2eb72](https://github.com/victorh-silveira/aether-quantum-engine/commit/3c2eb72bc9a31170eb4a36306e86148536bb5151))

## [1.7.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.6.0...v1.7.0) (2026-05-12)

### Funcionalidades

* **llm:** forca execucao em regimes de alta entropia ([e0b8f12](https://github.com/victorh-silveira/aether-quantum-engine/commit/e0b8f126df72d74193914e7ba77b61f1a3ed17da))

## [1.6.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.5.0...v1.6.0) (2026-05-12)

### Funcionalidades

* **llm:** relaxa filtros para aumentar frequencia ([767be0f](https://github.com/victorh-silveira/aether-quantum-engine/commit/767be0f992d6124bf70d3bdd74a37fe978ad4c65))

## [1.5.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.4.1...v1.5.0) (2026-05-12)

### Funcionalidades

* **llm:** melhora visualizacao de SKIP na linha de resposta ([2dcd80d](https://github.com/victorh-silveira/aether-quantum-engine/commit/2dcd80dc85ee35e9d8d23a4cbd03e78eb6fa14a9))

## [1.4.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.4.0...v1.4.1) (2026-05-12)

### Correcoes de Bug

* **llm:** diferencia skips de falhas de API nos logs ([db7c3e9](https://github.com/victorh-silveira/aether-quantum-engine/commit/db7c3e99b2540eefd3e259132a3be5fcb1e9e431))

## [1.4.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.3.0...v1.4.0) (2026-05-12)

### Funcionalidades

* **llm:** endurece precisao estatistica e otimiza latencia ([18f22a0](https://github.com/victorh-silveira/aether-quantum-engine/commit/18f22a0dbd7d1d89537841e184e89609d44b5bcd))

## [1.3.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.2.0...v1.3.0) (2026-05-12)

### Funcionalidades

* **risk:** atualiza stop win de banca pequena para 10 dolares ([1f79660](https://github.com/victorh-silveira/aether-quantum-engine/commit/1f7966029aba609fdce744ac28a78d7d1aeb02a6))

## [1.2.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.1.0...v1.2.0) (2026-05-12)

### Funcionalidades

* **llm:** endurece prompt contra superconfiança e otimiza densidade de dados ([3e30867](https://github.com/victorh-silveira/aether-quantum-engine/commit/3e30867a97370e21645c20b979d1b70d5608f39b))
* **llm:** otimização de custos da API Gemini ([c735ae8](https://github.com/victorh-silveira/aether-quantum-engine/commit/c735ae86ae4ee88603a8b80771ae44048281a1f3))

### Correcoes de Bug

* **llm:** resolve erro MAX_TOKENS aumentando limite para 128 ([f05ace0](https://github.com/victorh-silveira/aether-quantum-engine/commit/f05ace0f3ff4b069d667fc4e41774ab76e7e5e80))
* **llm:** restaura modelo gemini-2.5-flash após erro 404 ([9cab083](https://github.com/victorh-silveira/aether-quantum-engine/commit/9cab0837f734a3da1baebf70c78cd7a5b0d0de6b))

## [1.1.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v1.0.0...v1.1.0) (2026-05-12)

### Funcionalidades

* **risk:** recuperacao dinamica de 100% (Loss + Win Previsto) ([c42fa97](https://github.com/victorh-silveira/aether-quantum-engine/commit/c42fa97c6eaafe7862ed1a65cb41fa1f9407ef54))

### Documentacao

* **all:** reset de CHANGELOG para sincronizacao com Medallion 9.0 ([355ba41](https://github.com/victorh-silveira/aether-quantum-engine/commit/355ba416bfcbb3ca24d543fb2734c37d5e57cad5))

# Changelog

All notable changes to this project will be documented in this file.

## [9.0.0] (2026-05-12)

### Funcionalidades

* **all:** inicialização do motor Medallion 9.0 calibrado para R_100.
* **risk:** implementação de alvo de Stop Win de 3% e critério de Kelly conservador.
* **llm:** integração de motor soberano Gemini com lógica de inversão por convicção.

### Otimizações

* **engine:** recalibração de bins de entropia (30) e janelas de indicadores para mercado sintético.
* **telemetry:** logs segmentados para melhor visibilidade horizontal no terminal.
