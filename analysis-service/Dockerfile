FROM python:3.11-slim

WORKDIR /service

COPY requirements.txt .
# Install CPU-only torch first (saves ~1.5GB vs CUDA torch), then remaining deps
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app/ app/

RUN groupadd -r svcuser && useradd -r -g svcuser -m svcuser
RUN mkdir -p /service/data && chown -R svcuser:svcuser /service
USER svcuser

EXPOSE 8002

HEALTHCHECK --interval=10s --timeout=5s --retries=5 --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8002/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
