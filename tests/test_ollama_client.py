"""OllamaVLM against a tiny fake Ollama HTTP server (checks the request/response contract)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from src.inference.vlm import OllamaVLM, verify_crop

SEEN = {}


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5vl:3b"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.update(body)
        if body.get("images"):
            pass
        content = '{"verdict":"reject","label":"none","reason":"looks like a scratch"}'
        self._send({"message": {"role": "assistant", "content": content}})


def test_ollama_contract():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host = f"http://127.0.0.1:{srv.server_port}"
    try:
        vlm = OllamaVLM(host, "qwen2.5vl:3b", timeout=5)
        assert vlm.available()
        assert not OllamaVLM(host, "other-model:1b", timeout=5).available()
        out = verify_crop(vlm, np.zeros((64, 64, 3), np.uint8), "short")
        assert out["verdict"] == "reject" and out["provider"] == "ollama"
        assert SEEN["model"] == "qwen2.5vl:3b" and SEEN["stream"] is False and SEEN["format"] == "json"
        assert len(SEEN["messages"][0]["images"]) == 1  # base64 image attached
    finally:
        srv.shutdown()
    assert not OllamaVLM("http://127.0.0.1:9", "x", timeout=1).available()  # unreachable -> False, no crash
