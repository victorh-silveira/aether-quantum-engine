FROM python:3.13-slim
WORKDIR /opt/aether
COPY app/requirements.txt /opt/aether/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/aether/app/requirements.txt \
    && useradd --uid 10001 --create-home aether
COPY --chown=aether:aether app/src /opt/aether/app/src
COPY --chown=aether:aether app/run.py /opt/aether/app/run.py
COPY --chown=aether:aether config/settings.json /opt/aether/config/settings.json
ENV PYTHONUNBUFFERED=1
EXPOSE 9101
USER aether
CMD ["python", "app/run.py"]
