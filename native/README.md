# Modulos Nativos do Aether Engine (Rust)

Este diretorio abriga os componentes compilados de ultra-baixa latencia do Aether Quantum Engine, desenvolvidos em Rust para execucao no espaco de processo do host.

## Componentes

- **`aether_core`**: Crate em Rust compilado com PyO3 (`aether_core_rs`). Fornece ring buffer de potencia de 2 sem locks concorrentes para streaming de ticks e calculo sub-microssegundo de volatilidade realizada.

## Build e Testes

```bash
cargo test --manifest-path native/aether_core/Cargo.toml
```
