FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin aether

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY runtime.py learn_policy.py buffer_io.py app.py .

ENV MODELS_DIR=/models
ENV LOSS_FEATURE_DIM=24
ENV LOSS_READY_N=24
ENV LOSS_RETRAIN_MIN_N=1
ENV LOSS_RETRAIN_ON_LOSS_MIN_N=1
ENV LOSS_BOOTSTRAP_EXIT_N=16
ENV LOSS_MIN_WIN_FOR_LOSS_RETRAIN=1
ENV LOSS_MAX_BUFFER=2000
ENV LOSS_VETO_P_LOSS_FLOOR=0.65
EXPOSE 8000

USER aether

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=10 \
  CMD python -c "import json, urllib.request, sys; response = urllib.request.urlopen('http://127.0.0.1:8000/health'); payload = json.loads(response.read().decode()); sys.exit(0 if response.getcode() == 200 and payload.get('ready') else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
