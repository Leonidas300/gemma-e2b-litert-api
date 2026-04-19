# Gemma LiteRT API

OpenAI-compatible API server for **Gemma 4 E2B** running locally via [LiteRT-LM](https://ai.google.dev/edge/litert-lm).  
Optimized for **ARM64** (Oracle Cloud Free Tier, Raspberry Pi, etc.) and **low resource** environments.

Uses the official [LiteRT-LM Python API](https://ai.google.dev/edge/litert-lm/python) — the model loads once into memory and stays there, ensuring fast responses on every request.

## ✨ Features

- **OpenAI-compatible** — drop-in replacement for OpenAI SDK clients
- **Two API formats**:
  - `/v1/chat/completions` — classic OpenAI Chat API
  - `/v1/responses` — new OpenAI Responses API (used by n8n AI Agent, LangChain)
- **Streaming (SSE)** — supported in both endpoints
- **Lightweight** — debian:12-slim + FastAPI + uvicorn
- **ARM64 ready** — tested on Oracle Cloud Ampere A1

## 🚀 Quick Start

### 1. Download the model

On the host machine:

```bash
pip install huggingface_hub --break-system-packages
hf download litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --local-dir /opt/models/
```

Verify the file (~2.5 GB):

```bash
ls -lh /opt/models/
```

### 2. Run with docker-compose

```bash
git clone https://github.com/YOUR_USERNAME/gemma-litert-api
cd gemma-litert-api
docker compose up -d
```

First build takes 5–10 minutes (downloads Python dependencies).  
The model loads into memory ~30–60 seconds after container start.

### 3. Test

```bash
curl http://localhost:3008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "gemma-4-e2b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## 🔧 Portainer Deployment

**Stacks → Add stack → Repository**

- **Repository URL:** `https://github.com/YOUR_USERNAME/gemma-litert-api`
- **Repository reference:** `refs/heads/main`
- **Compose path:** `docker-compose.yml`

Click **Deploy the stack**.

## 📡 n8n Integration

Add **OpenAI API** credentials in n8n:

- **Base URL:** `http://litert:3000/v1` (if n8n is in the same Docker network)  
  or `http://HOST_IP:3008/v1`
- **API Key:** `sk-local`
- **Model:** `gemma-4-e2b`

Works with **AI Agent**, **Chat Model**, **HTTP Request** and other nodes. 
If you want to use tool than in n8n you need to import this agent https://github.com/fjrdomingues/n8n-nodes-better-ai-agent.git

## ⚙️ Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/gemma-4-E2B-it.litertlm` | Path to the model inside the container |
| `API_KEY` | `sk-local` | API key for Bearer authentication |
| `MODEL_ID` | `gemma-4-e2b` | Model ID returned by the API |
| `PORT` | `3000` | Internal container port |

## 📊 Resource Usage

No limits set by default. To restrict CPU/RAM, add to `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: "3.0"
      memory: 8G
```

The model itself uses ~3 GB RAM. Generation is CPU-intensive.

## 🛠️ API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Server health check |
| `GET`  | `/v1/models` | List available models |
| `POST` | `/v1/chat/completions` | Chat completions (classic OpenAI) |
| `POST` | `/v1/responses` | Responses API (new OpenAI format) |

## 📝 License

MIT

## 🙏 Acknowledgements

- [Google AI Edge](https://github.com/google-ai-edge/LiteRT-LM) — LiteRT-LM framework
- [Gemma](https://ai.google.dev/gemma) — models by Google DeepMind
