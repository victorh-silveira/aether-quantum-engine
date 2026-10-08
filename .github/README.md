# CI/CD da branch principal

O workflow `ci.yml` roda em push para `main` e PR destinado a essa branch. Valida Ruff, 100% de cobertura de linhas, Compose, dashboards e build Docker. Após push aprovado, publica imagem de prévia com tag do SHA no GHCR. PR não publica imagem. Release e deploy automáticos estão desativados.
