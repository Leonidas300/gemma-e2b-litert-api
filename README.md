# Gemma LiteRT API

OpenAI-compatible API dla **Gemma 4 E2B** działający lokalnie przez [LiteRT-LM](https://ai.google.dev/edge/litert-lm). 
Zoptymalizowany pod **ARM64** (Oracle Cloud Free Tier, Raspberry Pi, itp.) i **niskie zasoby**.

Używa oficjalnego [Python API LiteRT-LM](https://ai.google.dev/edge/litert-lm/python) — model ładuje się raz do pamięci i pozostaje tam, co zapewnia szybkie odpowiedzi.

## ✨ Funkcje

- **OpenAI-compatible** — drop-in replacement dla klientów OpenAI SDK
- **Dwa formaty API**:
  - `/v1/chat/completions` — klasyczne OpenAI
  - `/v1/responses` — nowe OpenAI (używane przez n8n AI Agent, LangChain)
- **Streaming (SSE)** — działa w obu endpointach
- **Lekki** — debian:12-slim + FastAPI + uvicorn
- **ARM64 ready** — testowane na Oracle Cloud Ampere A1

## 🚀 Szybki start (Docker)

### 1. Pobierz model

Na hoście (wymaga Python):

```bash
pip install huggingface_hub --break-system-packages
# Nowsza wersja używa polecenia 'hf'
hf download litert-community/gemma-4-E2B-it-litert-lm \
  gemma-4-E2B-it.litertlm \
  --local-dir /opt/models/
```

Sprawdź czy plik się pobrał (~2.5 GB):

```bash
ls -lh /opt/models/
```

### 2. Uruchom przez docker-compose

```bash
git clone https://github.com/leonidas300/gemma-litert-api
cd gemma-litert-api
docker compose up -d
```

Pierwsze budowanie trwa 5-10 minut (pobiera zależności Python).  
Model ładuje się do pamięci ~30-60 sekund po starcie kontenera.

### 3. Test

```bash
curl http://localhost:3008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local" \
  -d '{
    "model": "gemma-4-e2b",
    "messages": [{"role": "user", "content": "Cześć!"}]
  }'
```

## 🔧 Deployment przez Portainer

**Stacks → Add stack → Repository**

- **Repository URL:** `https://github.com/leonidas300/gemma-litert-api`
- **Repository reference:** `refs/heads/main`
- **Compose path:** `docker-compose.yml`

Kliknij **Deploy the stack**.

## 📡 Integracja z n8n

W n8n dodaj credentials **OpenAI API** z własnym endpointem:

- **Base URL:** `http://litert:3000/v1` (jeśli n8n w tej samej sieci Docker)  
  lub `http://IP_HOSTA:3008/v1`
- **API Key:** `sk-local`
- **Model:** `gemma-4-e2b`

Działa z **AI Agent**, **Chat Model**, **HTTP Request** i innymi node'ami.

## ⚙️ Konfiguracja

Zmienne środowiskowe (w `docker-compose.yml`):

| Zmienna | Domyślna | Opis |
|---|---|---|
| `MODEL_PATH` | `/models/gemma-4-E2B-it.litertlm` | Ścieżka do modelu w kontenerze |
| `API_KEY` | `sk-local` | Klucz API (ustaw coś mocniejszego w produkcji) |
| `MODEL_ID` | `gemma-4-e2b` | ID modelu zwracane w API |
| `PORT` | `3000` | Port wewnątrz kontenera |

## 📊 Limity zasobów

Domyślnie brak limitów. Jeśli chcesz ograniczyć CPU/RAM, dodaj do `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: "3.0"
      memory: 8G
```

Sam model zużywa ~3 GB RAM, generowanie obciąża CPU.

## 🛠️ API Endpointy

| Metoda | Ścieżka | Opis |
|---|---|---|
| `GET`  | `/health` | Status serwera |
| `GET`  | `/v1/models` | Lista dostępnych modeli |
| `POST` | `/v1/chat/completions` | Chat completions (klasyczne OpenAI) |
| `POST` | `/v1/responses` | Responses API (nowe OpenAI) |

## 📝 Licencja

MIT

## 🙏 Podziękowania

- [Google AI Edge](https://github.com/google-ai-edge/LiteRT-LM) — LiteRT-LM
- [Gemma](https://ai.google.dev/gemma) — modele od Google DeepMind
- Inspirowane [imertz/litert-lm-api-server](https://github.com/imertz/litert-lm-api-server)
