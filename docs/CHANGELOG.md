## [5.7.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.6.0...v5.7.0) (2026-11-20)

### Funcionalidades

* **risk:** calibrar meta diaria para 4 porcento e desativar tiro unico ([be18c20](https://github.com/victorh-silveira/aether-quantum-engine/commit/be18c20fe04f08433f09445794bed82cf377e445))

## [5.6.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.5.1...v5.6.0) (2026-11-20)

### Funcionalidades

* **engine:** adicionar variante bimodal dual_sniper m15 e contexto cross-asset ([4bcf145](https://github.com/victorh-silveira/aether-quantum-engine/commit/4bcf14565bb52e8615b47539ddb5a329eb3cc401))

## [5.5.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.5.0...v5.5.1) (2026-11-20)

### Correcoes de Bug

* **engine:** exigir aceleracao macd no setup senior e alinhar telemetria otc ([70f85f6](https://github.com/victorh-silveira/aether-quantum-engine/commit/70f85f6cef318c2b6d33facdc91079834b4cc812))

## [5.5.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.4.0...v5.5.0) (2026-11-20)

### Funcionalidades

* **engine:** remover bloqueio horario de killzone e liberar mercados fechados otc ([e327541](https://github.com/victorh-silveira/aether-quantum-engine/commit/e3275418f66d15658e9b8a863db114482d08d61d))

## [5.4.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.3.2...v5.4.0) (2026-11-19)

### Funcionalidades

* **all:** exigir aceleracao macd no r6, teto 2.30 atr e liberar 12 ativos otc ([5ae8b76](https://github.com/victorh-silveira/aether-quantum-engine/commit/5ae8b76d6fef594db4b572c8e82c846740f8c32e))

## [5.3.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.3.1...v5.3.2) (2026-11-19)

### Correcoes de Bug

* **engine:** endurecer filtro doji/macd e calibrar sinais em tendencia m15 ([44dcfa1](https://github.com/victorh-silveira/aether-quantum-engine/commit/44dcfa1d7f732b31ef977d4a29020f6dff2bd100))

## [5.3.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.3.0...v5.3.1) (2026-11-18)

### Correcoes de Bug

* **infra:** fixar runner ubuntu-24.04 no pipeline ci para python 3.13.12 ([ea8cff3](https://github.com/victorh-silveira/aether-quantum-engine/commit/ea8cff323f7566014865095a70aa5c3357988e74))

### Refatoracoes Tecnicas

* **all:** consolidar modulos internos, imutabilidade de risco e imports do indicador ([eca6a6b](https://github.com/victorh-silveira/aether-quantum-engine/commit/eca6a6bda73254cc617b83a3521acc49271ed688))
* **all:** mover modulos da raiz para src e unificar pyproject ([f1e97cb](https://github.com/victorh-silveira/aether-quantum-engine/commit/f1e97cb14446ca790764beb4257e130391b6849a))

## [5.3.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.2.2...v5.3.0) (2026-11-17)

### Funcionalidades

* **all:** adicionar regime adaptativo de tendencia m15 ao setup institucional ([a64dc26](https://github.com/victorh-silveira/aether-quantum-engine/commit/a64dc26214e7324d3471d4ed01f90463c5521492))

## [5.2.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.2.1...v5.2.2) (2026-11-16)

### Correcoes de Bug

* **all:** propagar endpoint otlp de telemetria na inicializacao ([fb1114a](https://github.com/victorh-silveira/aether-quantum-engine/commit/fb1114aa9d0319b6987f9ff48905272df55f9c66))

## [5.2.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.2.0...v5.2.1) (2026-11-16)

### Correcoes de Bug

* **all:** corrigir exportacao otlp, auditoria m15 e logs de bloqueio ([776c8c7](https://github.com/victorh-silveira/aether-quantum-engine/commit/776c8c753c49cb5ed49dde410521b4bb394dfd7e))

## [5.2.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.1.1...v5.2.0) (2026-11-13)

### Funcionalidades

* **all:** endurecer filtros institucionais m15 e selecionar ativos otc ([05c70fa](https://github.com/victorh-silveira/aether-quantum-engine/commit/05c70fa4dcad21c4e6371a47b8b1fbaf3a5d99b1))

## [5.1.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.1.0...v5.1.1) (2026-11-13)

### Correcoes de Bug

* **all:** isolar identificador de rastreio por ciclo e corrigir auditoria m15 ([bf9a159](https://github.com/victorh-silveira/aether-quantum-engine/commit/bf9a1594ec9c28f61a5d85573de9bfb2e0ebb44e))

## [5.1.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.0.3...v5.1.0) (2026-11-12)

### Funcionalidades

* **all:** consolidar observabilidade m15, killzone e sync deriv ([11849a5](https://github.com/victorh-silveira/aether-quantum-engine/commit/11849a59e599478d144ebcf888c625da31743c0f))

## [5.0.3](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.0.2...v5.0.3) (2026-11-10)

### Correcoes de Bug

* **engine:** otimizar killzones canonicas e consistencia de ativos ([fa08bb8](https://github.com/victorh-silveira/aether-quantum-engine/commit/fa08bb80ee350379a1b8ff8f7f25d47ec97d054c))

## [5.0.2](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.0.1...v5.0.2) (2026-11-10)

### Correcoes de Bug

* **engine:** blindar killzones canonicas e teto de exaustao ([a9c17f8](https://github.com/victorh-silveira/aether-quantum-engine/commit/a9c17f89e92963b08ed1f27058e5ad15301c4c5f))

## [5.0.1](https://github.com/victorh-silveira/aether-quantum-engine/compare/v5.0.0...v5.0.1) (2026-11-10)

### Correcoes de Bug

* **engine:** vetar divergencia contraria de macd e calibrar ranking ([4de8f05](https://github.com/victorh-silveira/aether-quantum-engine/commit/4de8f05944656463c2f69fd7af49ad510f97314a))

## [5.0.0](https://github.com/victorh-silveira/aether-quantum-engine/compare/v4.40.2...v5.0.0) (2026-11-10)

### ⚠ BREAKING CHANGES

* **engine:** transicao formal de versao maior para 5.0.0 estabelecendo
a maturidade operacional definitiva.

### Funcionalidades

* **engine:** consolidar arquitetura institucional senior e marco 5.0.0 ([8ed4018](https://github.com/victorh-silveira/aether-quantum-engine/commit/8ed401812ebdc19b204bc3b8d22e34343fa5c204))

# Changelog

Todas as alteracoes notaveis do Aether Quantum Engine a partir da versao senior institucional serao documentadas neste arquivo.
