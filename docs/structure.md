# Estrutura do Monorepo Aether Quantum Engine

Inventário arquitetural completo de diretórios, módulos e serviços do repositório:

```
aether-quantum-engine/
├── .agents/                                # Harness unificado de agentes de IA agnóstico
│   ├── agents/                             # Personas especializadas de subagentes
│   ├── rules/                              # 24 regras operacionais (.mdc)
│   ├── skills/                             # 25 habilidades com runbooks (SKILL.md)
│   └── mcp.json                            # Configuração unificada de servidores MCP
├── config/                                 # Configurações SSOT (settings.json, python.json)
├── data/                                   # Checkpoints de modelos, estados locais
├── docs/                                   # Documentação técnica de arquitetura e trading
├── infra/                                  # Infraestrutura de suporte e orquestração Docker
│   └── docker/                             # Compose, Redis conf, Timescale SQL, Grafana
├── linters/                                # Configurações de commitlint e hooks git
├── native/                                 # Módulos de performance em Rust (aether_core)
├── scripts/                                # Scripts operacionais (wsl, operations, batch)
├── src/                                    # Todo o código-fonte da aplicação (monorepo unificado)
│   ├── domain/                             # Domínio puro: entidades, matemática, risco
│   ├── application/                        # Casos de uso, orquestrador, portas outbound
│   ├── infrastructure/                     # Adaptadores: Deriv WSS, Redis, TimescaleDB
│   ├── presentation/                       # Logging estruturado, terminal, telemetria
│   └── indicator/                          # Subsistema analítico desacoplado do indicador público
├── tests/                                  # Suite de testes com 100% de cobertura
│   ├── unit/                               # Testes unitários do motor
│   ├── integration/                        # Testes de integração
│   └── indicator/                          # Testes do indicador público de mercado
├── tools/                                  # Utilitários de inspeção e scripts de apoio
├── AGENTS.md                               # Ponto de entrada soberano para IAs e devs
├── Makefile                                # Alvos de automação do projeto
├── prompt-model.md                         # Contrato padrão de scaffold de prompts
├── pyproject.toml                          # Manifesto do projeto Python e ferramentas
└── README.md                               # Apresentação do projeto e instruções de setup
```
