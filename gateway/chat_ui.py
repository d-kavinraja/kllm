"""Chat UI: llama.cpp's own web UI (MIT, vendored in ui/vendor/llama-ui), served at /chat-ui/ on the relay.

The UI runs in llama-server "router" mode, so it shows a model picker. kllm answers the few calls it makes:
  GET  /chat-ui/props               router props (no model) or the props of the backend serving ?model=
  GET  /chat-ui/v1/models           kllm chat models with status loaded / loading / unloaded and vision flags
  GET  /chat-ui/models/sse          the current status on connect, then changes, plus a reload when the set changes
  POST /chat-ui/models/load         accepted and ignored: a model starts on its first message (autoscale), never on a pick
  POST /chat-ui/v1/chat/completions the relay's normal chat route (routing, failover, prompt cache affinity)
  GET  /chat-ui/slots               [] (kllm does not expose per-slot state)
Everything else the UI might ask for (stream resume, tools, MCP proxy) answers 404, which the UI treats as absent.
"""
import asyncio, json, pathlib, time

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import media, vision

DIR = (pathlib.Path(__file__).parent / "ui" / "vendor" / "llama-ui").resolve()
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml"}
SSE_SCAN_SECONDS = 3.0          # how often the status stream re-reads the relay state
SSE_POLL_SECONDS = 0.3          # ... while noticing a closed tab (or a new state) this often


def _gz(rel):
    """A vendored file, served still gzipped (every browser accepts gzip)."""
    f = (DIR / (rel + ".gz")).resolve()
    if DIR not in f.parents or not f.is_file():
        return JSONResponse({"error": {"code": 404, "message": "not found", "type": "not_found_error"}}, status_code=404)
    ext = pathlib.Path(rel).suffix
    cache = "no-cache" if ext == ".html" else "public, max-age=31536000, immutable"    # bundle names carry a hash
    return Response(f.read_bytes(), media_type=TYPES.get(ext, "application/octet-stream"),
                    headers={"Content-Encoding": "gzip", "Cache-Control": cache, "Vary": "Accept-Encoding"})


def _served(R):
    """{model id: loaded | loading} for every model serving or starting right now, under the name the UI lists.

    A backend records the key it was launched for, and that key is what /v1/models must call the model: a backend
    whose model_key is an alias of, or was renamed since, a registry entry still has to line up with the entry a
    user picks, or the UI shows it as (not available).
    """
    s = R.state()
    health = s.get("health") or {}
    out = {}
    for p in (s.get("pending") or {}).values():
        key = R._canonical_model((p or {}).get("model_key"))
        if key:
            out.setdefault(key, "loading")
    for b in s.get("backends") or []:
        key = R._canonical_model(b.get("model_key"))
        if key and b.get("url") and (health.get(b["account"]) or {}).get("ok", True):
            out[key] = "loaded"
    return out


def chat_models(R):
    """[(key, model dict)] for chat models a user can pick (no media, no aliases)."""
    return [(k, m) for k, m in R._registry().items()
            if isinstance(m, dict) and media.kind(m) == "chat" and not m.get("alias_of") and not m.get("hidden")]


def _vision(key, m):
    tags = [str(t).lower() for t in (m.get("tags") or [])]
    return bool(m.get("mmproj")) or "vision" in tags or vision.is_vision_model(key)


def _aliases(R, key):
    """Other names this model is reachable by: registry aliases that point here, and raw backend model keys."""
    out = []
    for k, m in R._registry().items():
        if isinstance(m, dict) and m.get("alias_of") == key and k not in out:
            out.append(k)
    for b in R.state().get("backends") or []:
        mk = b.get("model_key")
        if mk and mk != key and R._canonical_model(mk) == key and mk not in out:
            out.append(mk)
    return out


def _row(R, key, m, status, aliases=()):
    now = int(time.time())
    row = {"id": key, "aliases": list(aliases), "tags": m.get("tags") or [], "object": "model", "owned_by": "kllm", "created": now,
           "status": {"value": status, "args": []},
           "architecture": {"input_modalities": ["text", "image"] if _vision(key, m) else ["text"], "output_modalities": ["text"]},
           "source": "preset", "can_remove": False}
    if m.get("ctx"):
        row["context_length"] = int(m["ctx"])
    return row


def _sse(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def stream_events(last, last_ids, data):
    """The SSE records that bring a status stream client from (last statuses, last_ids) to this payload.

    A fresh connection (last is None) is told every status: the page fetched /v1/models once, before it
    connected, and that answer may already be stale. A changed list of models needs a reload first, so the
    page picks up a backend that appeared or went away instead of waiting for a status event about a model
    it has never heard of.
    """
    ids = [row["id"] for row in data]
    now = {row["id"]: row["status"]["value"] for row in data}
    out = []
    if last is not None and ids != last_ids:
        out.append(("models_reload", None, None))
        last = None                              # the list changed: resend every status with it
    for key in ids:
        if last is None or last.get(key) != now[key]:
            out.append(("status_change", key, now[key]))
    return out


def models_payload(R):
    """Every model the UI may pick, each with the status of the backend actually serving it.

    Registry chat models come first. A backend can also be serving a model the registry no longer lists (models.yaml
    edited, or launched under another name): it is running and usable, so it is listed too - otherwise the UI cannot
    select it and every reply from it looks like it came from a model that is not available.
    """
    served = _served(R)
    reg = R._registry()
    data, listed = [], set()
    for key, m in chat_models(R):
        data.append(_row(R, key, m, served.get(key, "unloaded"), _aliases(R, key)))
        listed.add(key)
    for key, status in served.items():
        if key not in listed and key not in reg:
            data.append(_row(R, key, {}, status))
            listed.add(key)
    return {"object": "list", "data": data}


def router_props(R):
    return {"role": "router", "max_instances": 0, "models_autoload": True,
            "model_alias": "kllm", "model_path": "none",
            "default_generation_settings": {"params": {}, "n_ctx": 0},
            "ui_settings": {}, "build_info": "kllm relay", "cors_proxy_enabled": False}


def fallback_props(key, m):
    """Props for a model nothing serves yet: enough for the UI to show its context size and image button."""
    return {"role": "model", "model_alias": key, "model_path": key,
            "default_generation_settings": {"params": {}, "n_ctx": int(m.get("ctx") or 0)},
            "modalities": {"vision": _vision(key, m), "audio": False}, "build_info": "kllm relay"}


def make_router(R):
    r = APIRouter()

    @r.get("/chat-ui")
    async def root_redirect():
        return Response(status_code=307, headers={"Location": "/chat-ui/"})

    @r.get("/chat-ui/")
    async def index():
        return _gz("index.html")

    @r.get("/chat-ui/_app/{path:path}")
    async def assets(path: str):
        return _gz("_app/" + path)

    @r.get("/chat-ui/props")
    async def props(model: str = ""):
        if not model:
            return router_props(R)
        key = R._canonical_model(model)
        m = R._registry().get(key)
        if not isinstance(m, dict):
            if key in _served(R):                   # running, but no longer in models.yaml: answer for it anyway
                return fallback_props(key, {})
            return JSONResponse({"error": {"code": 404, "message": f"model '{model}' not found", "type": "not_found_error"}}, status_code=404)
        for label, url, api_key, mk in R._pool():
            if R._canonical_model(mk) != key:
                continue
            try:
                async with httpx.AsyncClient(timeout=8) as c:
                    res = await c.get(url + "/props", headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
                if res.status_code == 200:
                    p = res.json()
                    p["model_alias"] = key        # the UI matches messages to models by this name
                    p["role"] = "model"
                    # kllm runs llama-server headless, so its /props says ui:false, and the vendored picker
                    # reads that as "hide this model": the running model then vanishes from the list and
                    # "Loaded models" stays empty even though /v1/models calls it loaded.
                    p["ui"] = True
                    return p
            except (httpx.HTTPError, ValueError):
                continue
        return fallback_props(key, m)

    @r.get("/chat-ui/v1/models")
    async def models():
        return models_payload(R)

    @r.get("/chat-ui/models/sse")
    async def models_sse(request: Request):
        async def gen():
            yield b": kllm\n\n"
            # a page fetches /v1/models once and never again, so whatever it got can be stale by the time it
            # connects (opened before the backend was ready). Ask it to refetch, then send the status snapshot.
            yield _sse({"event": "models_reload"})
            last = last_ids = None
            while not await request.is_disconnected():
                data = models_payload(R)["data"]
                for kind, key, status in stream_events(last, last_ids, data):
                    yield (_sse({"event": "models_reload"}) if kind == "models_reload"
                           else _sse({"model": key, "event": "status_change", "data": {"status": status}}))
                last = {row["id"]: row["status"]["value"] for row in data}
                last_ids = [row["id"] for row in data]
                for _ in range(max(1, int(SSE_SCAN_SECONDS / SSE_POLL_SECONDS))):   # look again every 3 s, but notice a closed tab sooner
                    if await request.is_disconnected():
                        return
                    await asyncio.sleep(SSE_POLL_SECONDS)
                yield b": ping\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    @r.post("/chat-ui/models/load")
    async def load(request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        key = R._canonical_model(str(body.get("model") or ""))
        if key not in dict(chat_models(R)) and key not in _served(R):
            return JSONResponse({"error": {"code": 404, "message": f"model '{body.get('model')}' not found", "type": "not_found_error"}}, status_code=404)
        # picking a model in the UI must not spend a GPU: nothing launches here. The first message for it goes through
        # the relay like any other request, and autoscale starts a kernel then (and the relay holds the request until
        # it serves). 2026-11-10: browsing three models in the picker launched three kernels.
        return {"success": True}

    @r.post("/chat-ui/models/unload")
    async def unload():
        # stopping kernels is Fleet's job: the chat UI never shuts a backend down
        return {"success": True}

    @r.get("/chat-ui/slots")
    async def slots():
        return []

    @r.get("/chat-ui/tools")
    async def tools():
        # llama-server's built-in file and shell tools are not offered: the relay is not a sandbox. MCP servers can
        # still be added in the UI's settings (kllm's own is at /mcp on this origin).
        return []

    @r.get("/chat-ui/sw.js")
    async def service_worker():
        # the UI registers an offline worker; a no-op one keeps the console clean and caches nothing stale
        return Response("self.addEventListener('install', () => self.skipWaiting());\n", media_type="text/javascript",
                        headers={"Cache-Control": "no-cache"})

    @r.api_route("/chat-ui/v1/chat/completions", methods=["POST"])
    async def completions(request: Request):
        return await R.proxy("v1/chat/completions", request)

    @r.api_route("/chat-ui/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def absent(path: str):
        return JSONResponse({"error": {"code": 404, "message": f"kllm: /{path} is not available here", "type": "not_found_error"}}, status_code=404)

    return r
