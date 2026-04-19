"""
server.py – OpenAI-compatible API for Gemma 4 E2B via LiteRT-LM Python API.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/chat/completions  (classic OpenAI + tool calling)
  POST /v1/responses         (new OpenAI Responses API + tool calling)

Tool calling strategy:
  Prompt-engineering fallback. The model is instructed to output a specific
  JSON format when it wants to call a tool. The server parses this output
  and converts it into OpenAI's native tool_calls format so clients
  (n8n AI Agent, LangChain, etc.) can use it transparently.

  Special handling for n8n V3 Agent's internal tool: format_final_json_response.
  When the model calls this tool, we extract the output and return it as a
  normal text response — n8n uses this tool to format final answers.
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
app = FastAPI(title="Gemma 4 E2B API", version="4.0.0", lifespan=lifespan)
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

# ── Helpers ───────────────────────────────────────────────────
def message_content_to_text(content: Any) -> str:
    """Extract plain text from content field (string or list)."""
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

def build_tools_prompt(tools: list) -> str:
    """Build a system prompt describing available tools."""
    if not tools:
        return ""

    tool_descriptions = []
    for t in tools:
        fn = t.get("function", t) if isinstance(t, dict) else t
        name = fn.get("name", "unknown")
        # Skip n8n internal tools from description
        if name == "format_final_json_response":
            continue
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []

        param_lines = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "any") if isinstance(pinfo, dict) else "any"
            pdesc = pinfo.get("description", "") if isinstance(pinfo, dict) else ""
            req = " (required)" if pname in required else ""
            param_lines.append(f"    - {pname} ({ptype}){req}: {pdesc}")

        params_str = "\n".join(param_lines) if param_lines else "    (no parameters)"
        tool_descriptions.append(f"- {name}: {desc}\n  Parameters:\n{params_str}")

    if not tool_descriptions:
        return ""

    tools_text = "\n".join(tool_descriptions)

    return f"""You have access to the following tools:

{tools_text}

IMPORTANT — When you need to call a tool, respond with ONLY a JSON object in this exact format (no other text before or after):
{{"tool_call": {{"name": "tool_name", "arguments": {{"param1": "value1"}}}}}}

If you do NOT need to call a tool, respond normally in plain text.
After a tool is called and you receive its result, continue the conversation normally using that information."""

def parse_tool_call(text: str) -> Optional[dict]:
    """Try to parse a tool call from the model's raw output."""
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool_call" in data:
            tc = data["tool_call"]
            if "name" in tc:
                return {"name": tc["name"], "arguments": tc.get("arguments", {})}
    except json.JSONDecodeError:
        pass

    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
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
                    candidate = text[brace_start:i+1]
                    try:
                        data = json.loads(candidate)
                        if isinstance(data, dict) and "tool_call" in data:
                            tc = data["tool_call"]
                            if "name" in tc:
                                return {"name": tc["name"], "arguments": tc.get("arguments", {})}
                    except json.JSONDecodeError:
                        pass
                    break
        brace_start = text.find('{', brace_start + 1)

    return None

def extract_format_final_response(text: str) -> Optional[str]:
    """
    n8n V3 Agent adds 'format_final_json_response' tool internally.
    When the model calls it, extract the 'output' field and return as plain text.
    """
    text = text.strip()

    # Try to find tool_call with name format_final_json_response
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool_call" in data:
            tc = data["tool_call"]
            if tc.get("name") == "format_final_json_response":
                args = tc.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        pass
                return args.get("output", str(args))
    except json.JSONDecodeError:
        pass

    # Search in raw text
    patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "tool_call" in data:
                    tc = data["tool_call"]
                    if tc.get("name") == "format_final_json_response":
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                pass
                        return args.get("output", str(args))
            except json.JSONDecodeError:
                continue

    return None

def has_n8n_format_tool(tools: Optional[list]) -> bool:
    """Check if n8n injected its format_final_json_response tool."""
    if not tools:
        return False
    return any(
        (t.get("function", t) if isinstance(t, dict) else t).get("name") == "format_final_json_response"
        for t in tools
    )

def build_n8n_system_prompt(tools: Optional[list]) -> str:
    """
    Build system prompt for n8n V3 Agent mode.
    Instructs model to call format_final_json_response with its final answer.
    """
    user_tools = [
        t for t in (tools or [])
        if (t.get("function", t) if isinstance(t, dict) else t).get("name") != "format_final_json_response"
    ]

    tool_section = ""
    if user_tools:
        tool_section = build_tools_prompt(user_tools) + "\n\n"

    return f"""{tool_section}When you have a final answer for the user, you MUST respond with ONLY this JSON format:
{{"tool_call": {{"name": "format_final_json_response", "arguments": {{"output": "your answer here"}}}}}}

Replace "your answer here" with your actual response. Do not add any text before or after this JSON."""

def build_litertlm_messages(messages: list, tools: Optional[list] = None, n8n_mode: bool = False) -> tuple:
    """Convert OpenAI-style messages to litert_lm format."""
    if n8n_mode:
        tool_system = build_n8n_system_prompt(tools)
    else:
        tool_system = build_tools_prompt(tools) if tools else ""

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
                calls_text.append(json.dumps({"tool_call": {"name": name, "arguments": args}}))
            text = "\n".join(calls_text) if calls_text else text

        if role == "system":
            system_content = (system_content + "\n\n" + text).strip() if system_content else text
            continue

        result.append({
            "role": role,
            "content": [{"type": "text", "text": text}]
        })

    final_system = system_content
    if tool_system:
        final_system = (system_content + "\n\n" + tool_system).strip() if system_content else tool_system

    init_messages = []
    if final_system:
        init_messages.append({
            "role": "system",
            "content": [{"type": "text", "text": final_system}]
        })

    if result:
        init_messages.extend(result[:-1])
        last = result[-1]
        prompt_text = last["content"][0]["text"]
    else:
        prompt_text = ""

    return init_messages, prompt_text

def run_generation(init_messages: list, prompt_text: str) -> str:
    """Run full generation, return complete response text."""
    full_text = ""
    with engine.create_conversation(messages=init_messages) as conv:
        for chunk in conv.send_message_async(prompt_text):
            for item in chunk.get("content", []):
                if item.get("type") == "text":
                    full_text += item.get("text", "")
    return full_text

def stream_generation(init_messages: list, prompt_text: str):
    """Yield text tokens as they arrive."""
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
            "id": MODEL_ID, "object": "model", "created": 1700000000, "owned_by": "google",
        }]
    }

# ── /v1/chat/completions ──────────────────────────────────────
@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, _=Depends(verify_key)):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Detect n8n V3 Agent mode (it injects format_final_json_response)
    n8n_mode = has_n8n_format_tool(req.tools)

    init_messages, prompt_text = build_litertlm_messages(req.messages, req.tools, n8n_mode=n8n_mode)
    request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream and not req.tools:
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

    # n8n mode: check if model called format_final_json_response
    if n8n_mode:
        final_output = extract_format_final_response(full_text)
        if final_output:
            log.info("n8n format_final_json_response detected — returning as text")
            return JSONResponse({
                "id": request_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": final_output},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
            })

        # Check for user-defined tool call
        tool_call = parse_tool_call(full_text)
        if tool_call and tool_call["name"] != "format_final_json_response":
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
                            "function": {
                                "name": tool_call["name"],
                                "arguments": json.dumps(tool_call["arguments"])
                            }
                        }]
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
            })

        # Fallback: return raw text
        return JSONResponse({
            "id": request_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": full_text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1}
        })

    # Standard mode
    tool_call = parse_tool_call(full_text) if req.tools else None

    if tool_call:
        tc_id = f"call_{uuid.uuid4().hex[:24]}"
        return JSONResponse({
            "id": request_id, "object": "chat.completion", "created": created, "model": MODEL_ID,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tc_id, "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["arguments"])
                        }
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
    """SSE stream in chat.completion.chunk format."""
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
            yield f"data: {json.dumps({'error': {'message': item['__error__']}})}\n\n"
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

    init_messages, prompt_text = build_litertlm_messages(messages, req.tools)
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if req.stream and not req.tools:
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

    log.info(f"MODEL OUTPUT: {full_text[:200]}")
    tool_call = parse_tool_call(full_text) if req.tools else None

    if tool_call:
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        return JSONResponse({
            "id": response_id, "object": "response", "created_at": created,
            "status": "completed", "model": MODEL_ID,
            "output": [{
                "id": call_id, "type": "function_call", "status": "completed",
                "call_id": call_id,
                "name": tool_call["name"],
                "arguments": json.dumps(tool_call["arguments"])
            }],
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
    """SSE stream for Responses API."""
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
            "output": [{
                "id": msg_id, "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": full_text, "annotations": []}]
            }],
            "usage": {"input_tokens": -1, "output_tokens": -1, "total_tokens": -1}
        }
    }
    yield f"data: {json.dumps(done)}\n\n"
    yield "data: [DONE]\n\n"
