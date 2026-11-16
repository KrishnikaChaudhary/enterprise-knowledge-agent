import json
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from kagent.agent import Agent, AgentConfig
from kagent.embeddings import HashEmbedder
from kagent.ingest import ingest
from kagent.llm import LLMResult, MockLLM
from kagent.retrieval import Retriever
from kagent.store import SqliteVectorStore

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus"


@pytest.fixture(scope="session")
def embedder():
    return HashEmbedder()


@pytest.fixture(scope="session")
def store(embedder):
    s = SqliteVectorStore()
    ingest(CORPUS, s, embedder)
    return s


class SpyLLM:
    """Wraps MockLLM, records prompts, and can override a role's output or raise."""

    def __init__(self, overrides=None, raises=None):
        self.inner, self.overrides, self.raises, self.prompts = MockLLM(), overrides or {}, raises, []

    def complete(self, role, prompt, model):
        self.prompts.append((role, prompt))
        if self.raises and role in self.raises:
            raise self.raises[role]
        if role in self.overrides:
            out = self.overrides[role]
            text = out(len(self.prompts)) if callable(out) else out
            return LLMResult(text, 100, 10, model, 0.1, 0.5, 0.0001)
        return self.inner.complete(role, prompt, model)


@pytest.fixture
def make_agent(store, embedder):
    def _make(llm=None, **cfg):
        return Agent(Retriever(store, embedder), llm or MockLLM(), AgentConfig(**cfg))
    return _make


class FakeGemini:
    """Local stand-in for the Gemini REST API (AI Studio and Vertex shapes)."""

    def __init__(self, dim=8, fail_first=0):
        self.dim, self.fail_first, self.requests = dim, fail_first, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, dict(self.headers), body))
                if outer.fail_first > 0:
                    outer.fail_first -= 1
                    self.send_response(429)
                    self.end_headers()
                    return
                if ":streamGenerateContent" in self.path:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for obj in ({"candidates": [{"content": {"parts": [{"text": "Employees get 24 days "}]}}]},
                                {"candidates": [{"content": {"parts": [{"text": "[1]."}]}}], "usageMetadata": {"promptTokenCount": 321, "candidatesTokenCount": 9}}):
                        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                    return
                if ":batchEmbedContents" in self.path:
                    out = {"embeddings": [{"values": [1.0] + [0.0] * (outer.dim - 1)} for _ in body["requests"]]}
                elif ":predict" in self.path:
                    out = {"predictions": [{"embeddings": {"values": [0.0, 1.0] + [0.0] * (outer.dim - 2)}} for _ in body["instances"]]}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_port}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake_gemini():
    f = FakeGemini()
    yield f
    f.close()
