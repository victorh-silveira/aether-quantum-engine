FROM python:3.13-slim
WORKDIR /opt/aether
COPY app/requirements.txt /opt/aether/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/aether/app/requirements.txt
COPY app/src /opt/aether/app/src
COPY app/run.py /opt/aether/app/run.py
COPY config/settings.json /opt/aether/config/settings.json
ENV PYTHONUNBUFFERED=1
EXPOSE 9101
CMD ["python", "app/run.py"]
