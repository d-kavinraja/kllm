"""Regression: _pump must survive upstream silence longer than its keep-alive poll interval.

Bug (2026-10-31): _pump polled the upstream with asyncio.wait_for(it.__anext__(), 5.0). A timeout cancels the
pending __anext__, which finalises the httpx async generator, so the NEXT poll raised StopAsyncIteration and the
relay reported "backend ended the response early" (closing the upstream so llama-server cancelled the task) for any
request whose prefill was silent > 5 s after the response headers, i.e. every long-context agent turn.
"""
import asyncio, json
from gateway import relay


class FakeResp:
    status_code = 200
    headers = {"content-type": "text/event-stream"}

    def __init__(self, silence):
        self.silence = silence
        self.closed = False

    async def aiter_raw(self):
        yield b":\n\n"                                    # headers + SSE comment arrive immediately
        await asyncio.sleep(self.silence)                 # prefill: backend silent
        yield b'event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":9}}}\n\n'
        yield b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n\n'
        await asyncio.sleep(self.silence)                 # a second silent gap mid-stream
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

    async def aclose(self):
        self.closed = True


def _run(silence):
    relay.ROUTER.begin("t:silence")
    meter = relay._Meter("t:silence", "v1/messages", {"model": "m"})

    async def go():
        out = b""
        async for c in relay._pump(FakeResp(silence), meter, "v1/messages", "t:silence"):
            out += c
        return out
    return asyncio.run(go()), meter


def test_pump_survives_silence_longer_than_poll(monkeypatch):
    monkeypatch.setattr(relay, "PUMP_POLL_SECONDS", 0.05)
    out, meter = _run(silence=0.4)                        # 8 poll intervals of silence, twice
    assert meter.saw_end, "terminal event was never delivered"
    assert b"ended the response early" not in out
    assert b"message_stop" in out
    assert out.count(b": kllm: prefilling") >= 8          # keep-alive comments flowed while silent


def test_pump_no_silence_still_works(monkeypatch):
    monkeypatch.setattr(relay, "PUMP_POLL_SECONDS", 0.05)
    out, meter = _run(silence=0.0)
    assert meter.saw_end and b"ended the response early" not in out


def test_pump_names_the_routed_model_not_the_backends_own_name():
    """llama-server answers with its GGUF file name, which is not a /v1/models id: a UI that matches replies to
    pickable models by name (llama.cpp's picker) then shows the model as unavailable (2026-11-27)."""
    class Chatty:
        status_code = 200
        headers = {"content-type": "text/event-stream"}

        async def aiter_raw(self):
            yield b'data: {"id":"x","model":"Qwen--Qwen3.8-27B.Q4_K_M.gguf","choices":[]}\n\n'

        async def aclose(self):
            pass

    relay.ROUTER.begin("t:name")
    meter = relay._Meter("t:name", "v1/chat/completions", {"model": "qwen3.8-27b"})

    async def go():
        out = b""
        async for c in relay._pump(Chatty(), meter, "v1/chat/completions", "t:name", model_override="qwen3.8-27b"):
            out += c
        return out
    out = asyncio.run(go())
    assert b'"model":"qwen3.8-27b"' in out and b"Qwen--Qwen3.8" not in out
