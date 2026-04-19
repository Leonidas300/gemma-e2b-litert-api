"""
server.py – OpenAI-compatible API for Gemma 4 E2B via LiteRT-LM Python API.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions
  POST /v1/responses

Tool calling:
  Gemma 4 emits native tool calls using:
    <|tool_call>call:function_name{param:<|"|>value<|"|>}<tool_call|>
  We parse this format and convert to OpenAI tool_calls format.
  Also handles n8n's non-standard tool format (no "function" wrapper).
"""

import os
import re
import time
import json
import logging
import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Optional, Any

import litert_lm
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Configuration ────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/gemma-4-E2B-it.litertlm")
API_KEY    = os.environ.get("API_KEY", "sk-local")
MODEL_ID   = os.environ.get("MODEL_ID", "gemma-4-e2b")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gemma-api")

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

# ── Global engine ─────────────────────────────────────────────
engine: Optional[litert_lm.Engine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    log.info(f"Loading model: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}")
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU)
    log.info("Model loaded successfully ✓")
    yield
    if engine:
        try:
            engine.__exit__(None, None, None)
        except Exception:
            pass
    log.info("Engine closed")

# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(title="Gemma 4 E2B API", version="5.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Auth ──────────────────────────────────────────────────────
security = HTTPBearer(auto_error=False)

def verify_key(creds: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if API_KEY and (not creds or creds.credentials != API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API key")

# ── Pydantic models ───────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: Any = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    messages: list[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 1.0
    tools: Optional[list] = None
    tool_choice: Optional[Any] = None

class ResponsesRequest(BaseModel):
    model: Optional[str] = MODEL_ID
    input: Any
    stream: Optional[bool] = False
    tools: Optional[list] = None
    instructions: Optional[str] = None
    tool_choice: Optional[Any] = None

# ── Normalize tools ───────────────────────────────────────────
def normalize_tools(tools: Optional[list]) -> Optional[list]:
    """
    Normalize tool format. n8n sends tools without "function" wrapper:
      {"type": "function", "name": "...", "parameters": {...}}
    OpenAI format has wrapper:
      {"type": "function", "function": {"name": "...", "parameters": {...}}}
    """
    if not tools:
        return None
    normalized = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # Skip n8n internal tools
        if t.get("name") == "format_final_json_response":
            continue
        if t.get("function", {}).get("name") == "format_final_json_response":
            continue
        # n8n format: no "function" wrapper
        if "name" in t and "function" not in t:
            normalized.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("parameters", {"type": "object", "properties": {}})
                }
            })
        else:
            normalized.append(t)
    return normalized if normalized else None

# ── Gemma 4 native tool call parser ──────────────────────────
def parse_gemma4_tool_calls(text: str) -> Optional[list]:
    """
    Parse Gemma 4 native tool call format:
      <|tool_call>call:function_name{param:<|"|>value<|"|>}<tool_call|>
    """
    pattern = r'<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>'
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        return None

    tool_calls = []
    for func_name, args_str in matches:
        args_str = args_str.replace('<|"|>', '"')
        try:
            if not args_str.strip().startswith('{'):
                args_str = '{' + args_str + '}'
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            arguments = {}
            kv_pattern = r'(\w+):"([^"]*)"'
            for k, v in re.findall(kv_pattern, args_str):
                arguments[k] = v
            kv_pattern2 = r'(\w+):([\d.]+|true|false|null)'
            for k, v in re.findall(kv_pattern2, args_str):
                if v == 'true':
                    arguments[k] = True
                elif v == 'false':
                    arguments[k] = False
                elif v == 'null':
                    arguments[k] = None
                else:
                    try:
                        arguments[k] = float(v) if '.' in v else int(v)
                    except ValueError:
                        arguments[k] = v

        tool_calls.append({
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": func_name,
                "arguments": json.dumps(arguments)
            }
        })

    return tool_calls if tool_calls else None

def extract_text_without_tool_calls(text: str) -> str:
    """Remove Gemma 4 tool call blocks from text."""
    pattern = r'<\|tool_call>.*?<tool_call\|>'
    return re.sub(pattern, '', text, flags=re.DOTALL).strip()

def parse_tool_call_legacy(text: str) -> Optional[dict]:
    """Legacy JSON-based tool call parser."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool_call" in data:
            tc = data["tool_call"]
            if "name" in tc:
                return {"name": tc["name"], "arguments": tc.get("arguments", {})}
    except json.JSONDecodeError:
        pass

    for pattern in [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']:
        for match in re.findall(pattern, text, re.DOTALL):
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "tool_call" in data:
                    tc = data["tool_call"]
                    if "name" in tc:
                        return {"name": tc["name"], "arguments": tc.get("arguments", {})}
            except json.JSONDecodeError:
                continue

    brace_start = text.find('{')
    while brace_start != -1:
        depth = 0
        for i, c in enumerate(text[brace_start:], start=brace_start):
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[brace_start:i+1])
                        if isinstance(data, dict) and "tool_call" in data:
                            tc = data["tool_call"]
                            if "name" in tc:
                                return {"name": tc["name"], "arguments": tc.get("arguments", {})}
                    except json.JSONDecodeError:
                        pass
                    break
        brace_start = text.find('{', brace_start + 1)

    return None

# ── Helpers ───────────────────────────────────────────────────
def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
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

def build_tools_system_prompt(tools: list) -> str:
    """Build system prompt describing available tools for Gemma 4."""
    if not tools:
        return ""

    lines = ["You have access to the following tools:\n"]
    for t in tools:
        fn = t.get("function", t)
        name = fn.get("name", "")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []

        lines.append(f"- {name}: {desc}")
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any") if isinstance(pinfo, dict) else "any"
            pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
            req = " (required)" if pname in required else ""
            lines.append(f"    - {pname} ({ptype}){req}: {pdesc}")

    lines.append("""
When you need to call a tool, use this exact format:
<|tool_call>call:tool_name{<|"|>param<|"|>:<|"|>value<|"|>}<tool_call|>

If you don't need a tool, respond normally in plain text.""")

    return "\n".join(lines)

def build_litertlm_messages(messages: list, tools: Optional[list] = None) -> tuple:
    """Convert OpenAI-style messages to litert_lm format."""
    result = []
    system_content = ""

    for m in messages:
        role = m.role
        text = message_content_to_text(m.content)

        if role == "tool":
            tool_name = m.name or "tool"
            text = f"[Tool result from {tool_name}]: {text}"
            role = "user"

        if role == "assistant" and m.tool_calls:
            calls_text = []
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                args_gemma = json.dumps(args).replace('"', '<|"|>')
                calls_text.append(f'<|tool_call>call:{name}{args_gemma}<tool_call|>')
            text = "\n".join(calls_text) if calls_text else text

        if role == "system":
            system_content = (system_content + "\n\n" + text).strip() if system_content else text
            continue

        result.append({
            "role": role,
            "content": [{"type": "text", "text": text}]
        })

    # Add tools description to system prompt
    if tools:
        tools_prompt = build_tools_system_prompt(tools)
        if tools_prompt:
            system_content = (system_content + "\n\n" + tools_prompt).strip() if system_content else tools_prompt

    init_messages = []
    if system_content:
        init_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_content}]
        })

    if result:
        init_messages.extend(result[:-1])
        last = result[-1]
        prompt_text = last["content"][0]["text"]
    else:
        prompt_text = ""

    return init_messages, prompt_text

def run_generation(init_messages: list, prompt_text: str) -> str:
    full_text = ""
    with engine.create_conversation(messages=init_messages) as conv:
        for chunk in conv.send_message_async(prompt_text):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    full_text += item.get("text", "")
    return full_text

def stream_generation(init_messages: list, prompt_text: str):
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

@app.get("/v1/models")
def list_models(_=Depends(verify_key)):
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "created": 1700000000, "owned_by": "google"}]
    }

# ── /v1/chat/completions ──────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, _=Depends(verify_key)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    tools = normalize_tools(req.tools)
    init_messages, prompt_text = build_litertlm_messages(req.messages, tools)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream and not tools:
        return StreamingResponse(
            chat_stream_generator(request_id, created, init_messages, prompt_text),
            media_type="text/event-stream",
        )

    loop = asyncio.get_event_loop()
    try:
        full_text = await loop.run_in_executor(None, run_generation, init_messages, prompt_text)
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    log.info(f"MODEL OUTPUT: {full_text[:300]}")

    if tools:
        native_tool_calls = parse_gemma4_tool_calls(full_text)
        if native_tool_calls:
            log.info(f"Native tool calls: {[tc['function']['name'] for tc in native_tool_calls]}")
            return JSONResponse({
                "id": request_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": extract_text_without_tool_calls(full_text) or "",
                        "tool_calls": native_tool_calls
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
            })

        legacy_tc = parse_tool_call_legacy(full_text)
        if legacy_tc:
            tc_id = f"call_{uuid.uuid4().hex[:24]}"
            return JSONResponse({
                "id": request_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": tc_id, "type": "function",
                            "function": {"name": legacy_tc["name"], "arguments": json.dumps(legacy_tc["arguments"])}
                        }]
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
            })

    return JSONResponse({
        "id": request_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": full_text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
    })

async def chat_stream_generator(request_id, created, init_messages, prompt_text):
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

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, dict) and "__error__" in item:
            yield f"data: {json.dumps({'error': {'message': item['__error__']}})}\n\n"
            return
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {'content': item}, 'finish_reason': None}]})}\n\n"

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': created, 'model': MODEL_ID, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"

# ── /v1/responses ─────────────────────────────────────────────
@app.post("/v1/responses")
async def responses_endpoint(req: ResponsesRequest, _=Depends(verify_key)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = []
    if req.instructions:
        messages.append(Message(role="system", content=req.instructions))

    if isinstance(req.input, str):
        messages.append(Message(role="user", content=req.input))
    elif isinstance(req.input, list):
        for item in req.input:
            if isinstance(item, dict):
                messages.append(Message(
                    role=item.get("role", "user"),
                    content=item.get("content", ""),
                    tool_calls=item.get("tool_calls"),
                    name=item.get("name"),
                    tool_call_id=item.get("tool_call_id"),
                ))

    tools = normalize_tools(req.tools)
    init_messages, prompt_text = build_litertlm_messages(messages, tools)
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream and not tools:
        return StreamingResponse(
            responses_stream_generator(response_id, created, init_messages, prompt_text),
            media_type="text/event-stream",
        )

    loop = asyncio.get_event_loop()
    try:
        full_text = await loop.run_in_executor(None, run_generation, init_messages, prompt_text)
    except Exception as e:
        log.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    log.info(f"MODEL OUTPUT (responses): {full_text[:300]}")

    if tools:
        native_tool_calls = parse_gemma4_tool_calls(full_text)
        if native_tool_calls:
            log.info(f"Native tool calls in responses: {[tc['function']['name'] for tc in native_tool_calls]}")
            empty_msg_id = f"msg_{uuid.uuid4().hex[:24]}"
            return JSONResponse({
                "id": response_id, "object": "response", "created_at": created,
                "status": "completed", "model": MODEL_ID,
                "output": [
                    {
                        "id": empty_msg_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": []
                    }
                ] + [{
                    "id": tc["id"], "type": "function_call", "status": "completed",
                    "call_id": tc["id"],
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"]
                } for tc in native_tool_calls],
                "usage": {"input_tokens": -1, "output_tokens": -1, "total_tokens": -1}
            })

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    return JSONResponse({
        "id": response_id, "object": "response", "created_at": created,
        "status": "completed", "model": MODEL_ID,
        "output": [{
            "id": msg_id, "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": full_text, "annotations": []}]
        }],
        "usage": {"input_tokens": -1, "output_tokens": -1, "total_tokens": -1}
    })

async def responses_stream_generator(response_id, created, init_messages, prompt_text):
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

    yield f"data: {json.dumps({'type': 'response.created', 'response': {'id': response_id, 'object': 'response', 'created_at': created, 'model': MODEL_ID, 'status': 'in_progress'}})}\n\n"
    yield f"data: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'status': 'in_progress', 'content': []}})}\n\n"
    yield f"data: {json.dumps({'type': 'response.content_part.added', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': '', 'annotations': []}})}\n\n"

    full_text = ""
    while True:
        item = await queue.get()
        if item is None:
            break
        if isinstance(item, dict) and "__error__" in item:
            yield f"data: {json.dumps({'type': 'error', 'error': {'message': item['__error__']}})}\n\n"
            return
        full_text += item
        yield f"data: {json.dumps({'type': 'response.output_text.delta', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'delta': item})}\n\n"

    yield f"data: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'text': full_text})}\n\n"
    yield f"data: {json.dumps({'type': 'response.content_part.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': full_text, 'annotations': []}})}\n\n"
    yield f"data: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'status': 'completed', 'content': [{'type': 'output_text', 'text': full_text, 'annotations': []}]}})}\n\n"

    done = {
        "type": "response.completed",
        "response": {
            "id": response_id, "object": "response", "created_at": created,
            "status": "completed", "model": MODEL_ID,
            "output": [{"id": msg_id, "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": full_text, "annotations": []}]}],
            "usage": {"input_tokens": -1, "output_tokens": -1, "total_tokens": -1}
        }
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"
