# Estrutura do Monorepo Aether Quantum Engine

Inventário arquitetural completo de diretórios, módulos e serviços do repositório:

```
aether-quantum-engine/
├── .agents/                                # Harness unificado de agentes de IA
│   ├── agents/                             # Personas especializadas de subagentes
│   ├── rules/                              # 24 regras operacionais (.mdc)
│   ├── skills/                             # 25 habilidades com runbooks (SKILL.md)
│   └── mcp.json                            # Configuração unificada de servidores MCP
├── .cursor/                                # Camada de compatibilidade para Cursor IDE
├── app/                                    # Código-fonte principal do motor de trading
│   ├── src/
│   │   ├── domain/                         # Domínio puro: entidades, matemática, risco
│   │   ├── application/                    # Casos de uso, orquestrador, portas outbound
│   │   ├── infrastructure/                 # Adaptadores: Deriv WSS, Redis, TimescaleDB
│   │   └── presentation/                   # Logging estruturado, terminal, telemetria
│   ├── scripts/                            # Scripts operacionais (wsl, operations, batch)
│   └── tests/                              # Suite de testes com 100% de cobertura
├── config/                                 # Configurações SSOT (settings.json, python.json)
├── data/                                   # Checkpoints de modelos, estados locais
├── docs/                                   # Documentação técnica de arquitetura e trading
├── indicator/                              # Subsistema desacoplado do indicador público
│   ├── app/                                # Código do serviço analítico de mercado
│   ├── config/                             # Configurações do indicador
│   ├── data/                               # Modelos do indicador
│   ├── docs/                               # Documentação específica do indicador
│   └── infra/                              # Dashboards Grafana para sintéticos
├── infra/                                  # Infraestrutura de suporte e orquestração Docker
│   └── docker/                             # Compose, Redis conf, Timescale SQL, Grafana
├── linters/                                # Configurações de commitlint e hooks git
├── native/                                 # Módulos de performance em Rust (aether_core)
├── tools/                                  # Utilitários de inspeção e scripts de apoio
├── AGENTS.md                               # Ponto de entrada soberano para IAs e devs
├── Makefile                                # Alvos de automação do projeto
├── prompt-model.md                         # Contrato padrão de scaffold de prompts
└── README.md                               # Apresentação do projeto e instruções de setup
```
