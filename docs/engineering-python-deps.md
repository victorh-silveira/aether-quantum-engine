# Dependencias Python (anti-redundancia)

SSOT de pins: [`app/requirements.txt`](../app/requirements.txt), [`app/requirements-dev.txt`](../app/requirements-dev.txt), Docker `infra/docker/*/requirements.txt`.

Rule: `aether-python-deps.mdc`. Skill: `aether-python-deps`.

## Principios

1. Declarar todo pacote com `import` / `from` em codigo first-party no requirements do ambiente correspondente.
2. Nao pinar ferramenta so-transitiva ja coberta pelo wrapper (ex.: `coverage` sob `pytest-cov`).
3. Uma lib por papel; excecao documentada: dual-stack DataFrame.
4. Apos mudar `numpy` / `torch` / `scikit-learn`: smoke import + `pip check` no WSL antes do commit.

## Dual-stack DataFrame (intencional)

| Lib | Papel |
|-----|--------|
| `polars` | Features DL runtime (`dl_feature_indicators*`) |
| `pandas` | Meta tabular / LightGBM / scripts `train_meta_*` / Docker meta |

Proibido: misturar pandas e polars no mesmo modulo; adicionar 3a lib DF (`modin`, `dask`, `cudf`, etc.) sem mandato.

## joblib

Mantido como pin direto: `import joblib` em train meta/loss e containers. Nao remover por ser transitivo do sklearn.

## coverage

Nao listar em `requirements-dev.txt` se `pytest-cov` estiver presente (`pytest-cov` instala e gerencia `coverage`).

## numpy / torch / sklearn (ABI)

Pins atuais: `numpy==2.4.6`, `torch==2.10.0`, `scikit-learn==1.6.1`. Ao bump:

```bash
python -m pip check
python -c "import numpy, torch, sklearn, joblib, pandas, polars; print(numpy.__version__, torch.__version__)"
```

Falhas de C-extension / ARRAY_API = regressao de ABI; corrigir pins antes de merge.

## Onde editar

| Ambiente | Arquivo |
|----------|---------|
| Runtime / treino host | `app/requirements.txt` |
| Pre-commit / testes | `app/requirements-dev.txt` |
| Containers meta / loss | `infra/docker/meta-classifier/requirements.txt`, `infra/docker/loss-classifier/requirements.txt` |
