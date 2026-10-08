# CI/CD das duas branches

Workflow desta branch: [ci.yml](ci.yml).

A `main` conserva o motor anterior e seu pipeline original: Python, Docker, Shell,
Rust, workflows, release semântico e resumo. Push e PR destinados à `main`
executam esse pipeline.

A `codex/synthetic-indicator` conserva o indicador. O pipeline dessa branch tem a
mesma estrutura de áreas e etapas de validação do pipeline anterior. Os comandos
Python e Docker foram adaptados aos arquivos do indicador. Shell e Rust verificam
os componentes presentes; a entrada de código Rust exige configurar suas etapas
antes da integração. Push e PR destinados à branch executam as validações.
Somente um push aprovado publica uma imagem de prévia no GHCR, identificada pelo
SHA. PR não publica imagem. Não há release semântico nem deploy do indicador.
