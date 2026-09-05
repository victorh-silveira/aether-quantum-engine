FROM python:3.13-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin aether

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY runtime.py learn_policy.py buffer_io.py app.py ./

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    VECLIB_MAXIMUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    MODELS_DIR=/models \
    LOSS_FEATURE_DIM=24 \
    LOSS_READY_N=8 \
    LOSS_RETRAIN_MIN_N=1 \
    LOSS_RETRAIN_ON_LOSS_MIN_N=1 \
    LOSS_BOOTSTRAP_EXIT_N=8 \
    LOSS_MIN_WIN_FOR_LOSS_RETRAIN=1 \
    LOSS_MAX_BUFFER=2000 \
    LOSS_VETO_P_LOSS_FLOOR=0.65
EXPOSE 8000

USER aether

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=10 \
  CMD python -c "import json, urllib.request, sys; response = urllib.request.urlopen('http://127.0.0.1:8000/health'); payload = json.loads(response.read().decode()); sys.exit(0 if response.getcode() == 200 and payload.get('ready') else 1)"

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "uvloop", "--http", "httptools", "--timeout-keep-alive", "30"]
