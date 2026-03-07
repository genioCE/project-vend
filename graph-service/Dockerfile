FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

COPY app/ app/
COPY config/ config/

RUN groupadd -r svcuser && useradd -r -g svcuser -m svcuser
RUN mkdir -p /service/data && chown -R svcuser:svcuser /service
USER svcuser

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
