FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install \
        litert-lm-api-nightly \
        fastapi \
        "uvicorn[standard]" \
        python-multipart

WORKDIR /app
COPY server.py .

RUN mkdir -p /models

ENV MODEL_PATH=/models/gemma-4-E2B-it.litertlm \
    API_KEY=sk-local \
    MODEL_ID=gemma-4-e2b \
    HOST=0.0.0.0 \
    PORT=3000

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn server:app --host $HOST --port $PORT --workers 1"]
