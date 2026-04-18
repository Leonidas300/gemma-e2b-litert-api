"""
server.py – OpenAI-compatible API dla Gemma 4 E2B przez oficjalne LiteRT-LM Python API.

Endpointy:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions  (klasyczne OpenAI API - chat, streaming)
  POST /v1/responses          (nowe OpenAI API - używane przez n8n AI Agent)

Model ładuje się RAZ przy starcie i trzyma się w pamięci - szybkie odpowiedzi.
"""

import os
import time
import json
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Any

import litert_lm
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Konfiguracja ─────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/gemma-4-E2B-it.litertlm")
API_KEY    = os.environ.get("API_KEY", "sk-local")
MODEL_ID   = os.environ.get("MODEL_ID", "gemma-4-e2b")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gemma-api")

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

# ── Globalny engine ────────────────────────────────────────────
engine: Optional[litert_lm.Engine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    log.info(f"Ładowanie modelu: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model nie znaleziony: {MODEL_PATH}")
        raise RuntimeError(f"Model nie znaleziony: {MODEL_PATH}")

    engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU)
    log.info("Model załadowany pomyślnie ✓")
    yield
    if engine:
        try:
            engine.__exit__(None, None, None)
        except Exception:
            pass
    log.info("Engine zamknięty")

# ── Aplikacja FastAPI ─────────────────────────────────────────
app = FastAPI(title="Gemma 4 E2B API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ──────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

def verify_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if API_KEY and (not creds or creds.credentials != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

# ── Modele Pydantic ───────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: Any  # string albo list - dla kompatybilności

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: list[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 1.0
    tools: Optional[list] = None

class ResponsesRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    input: Any
    stream: Optional[bool] = False
    tools: Optional[list] = None
    instructions: Optional[str] = None

# ── Pomocnicze ────────────────────────────────────────────────
def message_content_to_text(content: Any) -> str:
    """Wyciąga tekst z pola content (może być string lub lista)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text", "output_text"):
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return " ".join(parts)
    return str(content)

def build_litertlm_messages(messages: list[Message]) -> list[dict]:
    """Konwertuje wiadomości OpenAI → format litert_lm."""
    result = []
    for m in messages:
        text = message_content_to_text(m.content)
        result.append({
            "role": m.role,
            "content": [{"type": "text", "text": text}]
        })
    return result

def run_generation(init_messages: list, prompt_text: str) -> str:
    """Uruchamia pełne generowanie i zwraca cały tekst odpowiedzi."""
    full_text = ""
    with engine.create_conversation(messages=init_messages) as conv:
        for chunk in conv.send_message_async(prompt_text):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    full_text += item.get("text", "")
    return full_text

def stream_generation(init_messages: list, prompt_text: str):
    """Generator yield'ujący kolejne tokeny."""
    with engine.create_conversation(messages=init_messages) as conv:
        for chunk in conv.send_message_async(prompt_text):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        yield text

# ── Health ────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "engine_ready": engine is not None}

# ── /v1/models ────────────────────────────────────────────────
@app.get("/v1/models")
def list_models(_=Depends(verify_key)):
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 1700000000,
            "owned_by": "google",
        }]
    }

# ── /v1/chat/completions ──────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, _=Depends(verify_key)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model nie załadowany")

    lm_messages = build_litertlm_messages(req.messages)

    # Rozdziel system + historia + ostatnia wiadomość
    init_messages = []
    system_found = False
    for m in lm_messages:
        if m["role"] == "system" and not system_found:
            init_messages.append(m)
            system_found = True

    # Wszystkie wiadomości user/assistant oprócz ostatniej user
    non_system = [m for m in lm_messages if m["role"] != "system"]
    if non_system:
        init_messages.extend(non_system[:-1])
        last = non_system[-1]
    else:
        last = {"content": [{"text": ""}]}

    prompt_text = last["content"][0]["text"] if last.get("content") else ""
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            chat_stream_generator(request_id, created, init_messages, prompt_text),
            media_type="text/event-stream",
        )

    loop = asyncio.get_event_loop()
    try:
        full_text = await loop.run_in_executor(None, run_generation, init_messages, prompt_text)
    except Exception as e:
        log.error(f"Błąd generowania: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse({
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
    })

async def chat_stream_generator(request_id, created, init_messages, prompt_text):
    """SSE stream w formacie OpenAI chat.completion.chunk."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        try:
            for text in stream_generation(init_messages, prompt_text):
                asyncio.run_coroutine_threadsafe(queue.put(text), loop)
        except Exception as e:
            log.error(f"Producer error: {e}")
            asyncio.run_coroutine_threadsafe(queue.put({"__error__": str(e)}), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop.run_in_executor(None, producer)

    # Pierwszy chunk - role
    first = {
        "id": request_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
    }
    yield f"data: {json.dumps(first)}\n\n"

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, dict) and "__error__" in item:
            err = {"error": {"message": item["__error__"], "type": "internal_error"}}
            yield f"data: {json.dumps(err)}\n\n"
            return
        chunk = {
            "id": request_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
            "choices": [{"index": 0, "delta": {"content": item}, "finish_reason": None}]
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    final = {
        "id": request_id, "object": "chat.completion.chunk", "created": created, "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    }
    yield f"data: {json.dumps(final)}\n\n"
    yield "data: [DONE]\n\n"

# ── /v1/responses (n8n AI Agent używa tego) ──────────────────
@app.post("/v1/responses")
async def responses_endpoint(req: ResponsesRequest, _=Depends(verify_key)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model nie załadowany")

    # Zbuduj wiadomości z input (może być string albo lista)
    init_messages = []
    if req.instructions:
        init_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": req.instructions}]
        })

    prompt_text = ""
    if isinstance(req.input, str):
        prompt_text = req.input
    elif isinstance(req.input, list):
        # n8n wysyła listę wiadomości
        msgs = []
        for item in req.input:
            if isinstance(item, dict):
                role = item.get("role", "user")
                content = message_content_to_text(item.get("content", ""))
                msgs.append({"role": role, "content": content})
        if msgs:
            for m in msgs[:-1]:
                init_messages.append({
                    "role": m["role"],
                    "content": [{"type": "text", "text": m["content"]}]
                })
            prompt_text = msgs[-1]["content"]

    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream:
        return StreamingResponse(
            responses_stream_generator(response_id, created, init_messages, prompt_text),
            media_type="text/event-stream",
        )

    loop = asyncio.get_event_loop()
    try:
        full_text = await loop.run_in_executor(None, run_generation, init_messages, prompt_text)
    except Exception as e:
        log.error(f"Błąd generowania: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return JSONResponse({
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "model": MODEL_ID,
        "output": [{
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": full_text, "annotations": []}]
        }],
        "usage": {"input_tokens": -1, "output_tokens": -1, "total_tokens": -1}
    })

async def responses_stream_generator(response_id, created, init_messages, prompt_text):
    """SSE stream w formacie OpenAI responses API."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def producer():
        try:
            for text in stream_generation(init_messages, prompt_text):
                asyncio.run_coroutine_threadsafe(queue.put(text), loop)
        except Exception as e:
            asyncio.run_coroutine_threadsafe(queue.put({"__error__": str(e)}), loop)
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop.run_in_executor(None, producer)

    # response.created
    yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': response_id, 'object': 'response', 'created_at': created, 'model': MODEL_ID, 'status': 'in_progress'}})}\n\n"

    # output_item.added
    yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'status': 'in_progress', 'content': []}})}\n\n"

    # content_part.added
    yield f"data: {json.dumps({'type': 'response.content_part.added', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})}\n\n"

    full_text = ""
    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, dict) and "__error__" in item:
            err = {"type": "error", "error": {"message": item["__error__"]}}
            yield f"data: {json.dumps(err)}\n\n"
            return
        full_text += item
        delta = {
            "type": "response.output_text.delta",
            "item_id": msg_id,
            "output_index": 0,
            "content_index": 0,
            "delta": item
        }
        yield f"data: {json.dumps(delta)}\n\n"

    # Końcowe zdarzenia
    yield f"data: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'text': full_text})}\n\n"
    yield f"data: {json.dumps({'type': 'response.content_part.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': full_text, 'annotations': []}})}\n\n"
    yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': full_text, 'annotations': []}]}})}\n\n"

    done = {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "created_at": created,
            "status": "completed",
            "model": MODEL_ID,
            "output": [{
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_text, "annotations": []}]
            }],
            "usage": {"input_tokens": -1, "output_tokens": -1, "total_tokens": -1}
        }
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"
