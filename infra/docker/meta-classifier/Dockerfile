FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin aether

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py learn_runtime.py .

ENV MODELS_DIR=/models
ENV META_RETRAIN_MIN_N=2
ENV META_MAX_BUFFER=2000
EXPOSE 8000

USER aether

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=10 \
  CMD python -c "import urllib.request, sys; r=urllib.request.urlopen('http://127.0.0.1:8000/health'); sys.exit(0 if r.getcode()==200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
