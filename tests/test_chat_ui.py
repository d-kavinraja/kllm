"""The vendored llama.cpp chat UI on /chat-ui/: kllm answers its router-mode calls from the relay's own state."""
import json
from types import SimpleNamespace

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway import chat_ui

REG = {"qwen": {"ctx": 262144, "tags": ["vision"]}, "small": {"ctx": 32768}, "img": {"runtime": "diffusers", "kind": "image"},
       "alias": {"alias_of": "qwen"}}


def fake_relay(spawned, backends=(), pending=None, pool=()):
    st = {"backends": list(backends), "pending": pending or {}, "health": {}}
    return SimpleNamespace(state=lambda: st, _registry=lambda: REG, _canonical_model=lambda k: (REG.get(k) or {}).get("alias_of", k),
                           _pool=lambda: list(pool), _spawn=lambda args: spawned.append(args))


def client(r):
    app = FastAPI()
    app.include_router(chat_ui.make_router(r))
    return TestClient(app)


def test_models_list_chat_models_with_live_status():
    r = fake_relay([], backends=[{"account": "kg1", "url": "https://x", "model_key": "qwen"}], pending={"kg2": {"model_key": "small"}})
    data = {m["id"]: m for m in client(r).get("/chat-ui/v1/models").json()["data"]}
    assert set(data) == {"qwen", "small"}                                   # no image models, no aliases
    assert data["qwen"]["status"]["value"] == "loaded" and data["small"]["status"]["value"] == "loading"
    assert data["qwen"]["architecture"]["input_modalities"] == ["text", "image"] and data["qwen"]["context_length"] == 262144


def test_picking_a_model_never_launches_a_kernel():
    spawned = []
    c = client(fake_relay(spawned))
    assert c.post("/chat-ui/models/load", json={"model": "small"}).json() == {"success": True}
    assert spawned == []


def test_props_and_ui_files():
    c = client(fake_relay([]))
    assert c.get("/chat-ui/props").json()["role"] == "router"
    p = c.get("/chat-ui/props", params={"model": "qwen"}).json()          # nothing serves it: props from models.yaml
    assert p["modalities"]["vision"] and p["default_generation_settings"]["n_ctx"] == 262144
    r = c.get("/chat-ui/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert c.get("/chat-ui/tools").json() == [] and c.get("/chat-ui/nope").status_code == 404


def test_a_backend_serving_an_alias_is_loaded_under_the_model_a_user_picks():
    """The backend records the key it launched for, which can be an alias of the registry entry the picker lists:
    reporting it loaded only under the raw key leaves the entry a user picks looking unloaded (2026-11-27)."""
    r = fake_relay([], backends=[{"account": "kg1", "url": "https://x", "model_key": "alias"}])
    c = client(r)
    data = {m["id"]: m for m in c.get("/chat-ui/v1/models").json()["data"]}
    assert set(data) == {"qwen", "small"}                                   # no duplicate row for the alias
    assert data["qwen"]["status"]["value"] == "loaded"
    assert "alias" in data["qwen"]["aliases"]                               # ...but the alias still resolves here
    assert c.get("/chat-ui/props", params={"model": "alias"}).json()["model_alias"] == "qwen"


def test_a_loaded_backend_the_registry_dropped_is_still_listed_and_usable():
    """A backend outlives its models.yaml entry (renamed key, edited registry, launch under another name). It is
    running and every reply says so, so the picker has to offer it and props/load must answer for it."""
    r = fake_relay([], backends=[{"account": "kg1", "url": "https://x", "model_key": "qwen3.8-27b"}])
    c = client(r)
    data = {m["id"]: m for m in c.get("/chat-ui/v1/models").json()["data"]}
    assert set(data) == {"qwen", "small", "qwen3.8-27b"}
    assert data["qwen3.8-27b"]["status"]["value"] == "loaded"
    p = c.get("/chat-ui/props", params={"model": "qwen3.8-27b"}).json()
    assert p["model_alias"] == "qwen3.8-27b" and p["role"] == "model"    # not a 404 the UI shows as unavailable
    assert c.post("/chat-ui/models/load", json={"model": "qwen3.8-27b"}).json() == {"success": True}


def test_a_backend_with_its_own_web_ui_turned_off_still_lists_the_model(monkeypatch):
    """kllm starts llama-server headless, so its /props answers ui:false, and the vendored picker reads that
    as "hide this model": proxying it verbatim dropped the running model from the list and left "Loaded
    models" empty while /v1/models called it loaded (2026-11-27)."""
    asked = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, headers=None):
            asked["url"] = url
            return SimpleNamespace(status_code=200, json=lambda: {"model_path": "m.gguf", "ui": False,
                                                                "modalities": {"vision": False}})

    monkeypatch.setattr(chat_ui.httpx, "AsyncClient", FakeClient)
    r = fake_relay([], backends=[{"account": "kg1", "url": "https://x", "model_key": "qwen"}],
                   pool=[("kg1", "https://x", None, "qwen")])
    p = client(r).get("/chat-ui/props", params={"model": "qwen"}).json()
    assert asked["url"] == "https://x/props"
    assert p["ui"] is True                                              # the running model stays pickable
    assert p["model_alias"] == "qwen" and p["role"] == "model"          # the rest is still the backend's answer
    assert p["modalities"] == {"vision": False}


@pytest.mark.anyio
async def test_status_stream_opens_with_the_current_list_so_a_stale_page_catches_up(anyio_backend):
    """The page fetches /v1/models once: a stream opened later (backend came up meanwhile) must still hand it the
    list and the loaded status, without waiting for a status change that may never come."""
    if anyio_backend != "asyncio":
        pytest.skip("starlette's is_disconnected is asyncio-only here (uvicorn serves this app on asyncio)")
    app = FastAPI()
    app.include_router(chat_ui.make_router(fake_relay([], backends=[{"account": "kg1", "url": "https://x", "model_key": "qwen"}])))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET", "scheme": "http",
             "path": "/chat-ui/models/sse", "raw_path": b"/chat-ui/models/sse", "query_string": b"", "root_path": "",
             "headers": [(b"host", b"test")], "client": ("127.0.0.1", 50000), "server": ("test", 80)}
    sent = []

    async def receive():                       # TestClient buffers a stream until it ends; drive the app directly
        await anyio.sleep(3600)

    async def send(message):
        sent.append(message)

    async with anyio.create_task_group() as tg:
        tg.start_soon(app, scope, receive, send)
        for _ in range(300):
            await anyio.sleep(0.01)
            if any(b'"model": "qwen"' in m.get("body", b"") for m in sent if m.get("type") == "http.response.body"):
                break
        tg.cancel_scope.cancel()
    text = b"".join(m.get("body", b"") for m in sent if m.get("type") == "http.response.body").decode()
    events = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]
    assert events[0] == {"event": "models_reload"}            # refetch the list first, then the statuses
    assert {"model": "qwen", "event": "status_change", "data": {"status": "loaded"}} in events


def test_stream_events_cover_a_fresh_connection_a_change_and_a_new_model():
    rows = lambda *pairs: [{"id": k, "status": {"value": v}} for k, v in pairs]
    fresh = chat_ui.stream_events(None, None, rows(("a", "loaded"), ("b", "unloaded")))
    assert fresh == [("status_change", "a", "loaded"), ("status_change", "b", "unloaded")]
    assert chat_ui.stream_events({"a": "loaded", "b": "unloaded"}, ["a", "b"], rows(("a", "loaded"), ("b", "unloaded"))) == []
    assert chat_ui.stream_events({"a": "loading"}, ["a"], rows(("a", "loaded"))) == [("status_change", "a", "loaded")]
    # a model the page has never heard of: reload the list first, then send every status with it
    assert chat_ui.stream_events({"a": "loaded"}, ["a"], rows(("a", "loaded"), ("c", "loaded"))) == [
        ("models_reload", None, None), ("status_change", "a", "loaded"), ("status_change", "c", "loaded")]
