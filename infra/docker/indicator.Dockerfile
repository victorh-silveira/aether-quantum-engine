# ==============================================================================
#                 AETHER QUANTUM ENGINE - INDICATOR SERVICE
#                      ESTAGIO 1: BUILDER MULTI-STAGE
# ==============================================================================
FROM python:3.13-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY infra/docker/indicator-requirements.txt /build/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /build/requirements.txt

# ==============================================================================
#                      ESTAGIO 2: RUNNER MINIMO & SEGURO
# ==============================================================================
FROM python:3.13-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/opt/aether:/opt/aether/src"

WORKDIR /opt/aether

# Dependencias minimas de runtime: tini (PID 1), curl (healthcheck) e certificados
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Principio do Menor Privilegio (Non-Root)
RUN groupadd -g 10001 aether && \
    useradd -u 10001 -g aether -s /sbin/nologin -M aether && \
    mkdir -p /opt/aether/data/indicator-models && \
    chown -R aether:aether /opt/aether

# Copia ambiente Python isolado do builder
COPY --from=builder --chown=aether:aether /opt/venv /opt/venv

# Copia apenas os artefatos de aplicacao e configuracao necessarios
COPY --chown=aether:aether src /opt/aether/src
COPY --chown=aether:aether config /opt/aether/config
COPY --chown=aether:aether aether_paths.py /opt/aether/aether_paths.py
COPY --chown=aether:aether run_indicator.py /opt/aether/run_indicator.py

USER aether:aether

EXPOSE 9101

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "run_indicator.py"]
