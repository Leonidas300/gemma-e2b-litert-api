# gemma-e2b-litert-api
OpenAI-compatible API server for Gemma 4 E2B via LiteRT-LM. Supports /v1/chat/completions and /v1/responses (n8n AI Agent), SSE streaming, Bearer auth. Model loaded once in memory for fast responses. ARM64-optimized for Oracle Cloud Free Tier, Raspberry Pi. FastAPI + debian-slim. Deploy via docker-compose or Portainer.
