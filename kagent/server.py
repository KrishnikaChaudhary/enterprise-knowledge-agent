"""HTTP service (standard library only), shaped for Cloud Run: listens on $PORT, logs JSON to stdout,
drains in-flight requests on SIGTERM, exposes /healthz, /readyz and Prometheus /metrics.

Identity: the caller's ROLE decides which chunks the retriever may even score. It comes from the bearer
token (KB_TOKENS), never from the request body, unless KB_ALLOW_ROLE_IN_BODY=1 on a loopback host (dev/eval).
Behind Cloud Run IAM or IAP, map the verified identity to a role in your gateway and pass a token.
"""
from __future__ import annotations

import hmac
import json
import logging
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent import Agent, AgentConfig
from .errors import ConfigError
from .factory import build
from .settings import Settings

log = logging.getLogger("kagent")
ROLE_OK = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {"severity": record.levelname, "message": record.getMessage(), "logger": record.name}
        d.update(getattr(record, "fields", {}))
        return json.dumps(d, default=str)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests: dict[str, int] = {}
        self.tokens_in = self.tokens_out = 0
        self.cost = 0.0
        self.latency_sum = 0.0
        self.http_errors: dict[int, int] = {}

    def record(self, status: str, metrics: dict) -> None:
        with self._lock:
            self.requests[status] = self.requests.get(status, 0) + 1
            self.tokens_in += metrics.get("tokens_in", 0)
            self.tokens_out += metrics.get("tokens_out", 0)
            self.cost += metrics.get("cost_usd", 0.0)
            self.latency_sum += metrics.get("wall_s", 0.0)

    def http_error(self, code: int) -> None:
        with self._lock:
            self.http_errors[code] = self.http_errors.get(code, 0) + 1

    def render(self, chunks: int) -> str:
        with self._lock:
            lines = ["# TYPE kagent_requests_total counter"]
            lines += [f'kagent_requests_total{{status="{k}"}} {v}' for k, v in sorted(self.requests.items())]
            lines += ["# TYPE kagent_http_errors_total counter"] + [f'kagent_http_errors_total{{code="{k}"}} {v}' for k, v in sorted(self.http_errors.items())]
            lines += ["# TYPE kagent_tokens_total counter", f'kagent_tokens_total{{direction="in"}} {self.tokens_in}', f'kagent_tokens_total{{direction="out"}} {self.tokens_out}',
                      "# TYPE kagent_cost_usd_total counter", f"kagent_cost_usd_total {self.cost:.6f}",
                      "# TYPE kagent_request_seconds_sum counter", f"kagent_request_seconds_sum {self.latency_sum:.4f}",
                      "# TYPE kagent_indexed_chunks gauge", f"kagent_indexed_chunks {chunks}"]
        return "\n".join(lines) + "\n"


class App:
    def __init__(self, agent: Agent, settings: Settings, chunk_count):
        self.agent, self.settings, self.chunk_count = agent, settings, chunk_count
        self.metrics = Metrics()
        self.slots = threading.BoundedSemaphore(settings.max_concurrent)
        self.tokens = {t: r for t, r in settings.token_roles}

    def authenticate(self, header: str | None) -> tuple[bool, str | None]:
        """Returns (ok, role). With no token map configured, authentication is delegated to the platform (IAM)."""
        if not self.tokens:
            return True, None
        presented = header[7:].strip() if header and header.lower().startswith("bearer ") else ""
        role = None
        for tok, r in self.tokens.items():  # constant-time over every token: no early exit on a match
            if hmac.compare_digest(presented.encode(), tok.encode()):
                role = r
        return role is not None, role


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "kagent"

        def log_message(self, fmt, *args):  # route access logs through structured logging
            log.info("http", extra={"fields": {"client": self.client_address[0], "line": fmt % args}})

        def _send(self, code: int, obj, ctype: str = "application/json") -> None:
            body = obj.encode() if isinstance(obj, str) else json.dumps(obj).encode()
            if code >= 400:
                app.metrics.http_error(code)
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/healthz":
                return self._send(200, {"status": "ok"})
            if self.path == "/readyz":
                n = app.chunk_count()
                return self._send(200 if n else 503, {"ready": bool(n), "chunks": n})
            if self.path == "/metrics":
                ok, _ = app.authenticate(self.headers.get("Authorization"))
                if not ok:
                    return self._send(401, {"error": "unauthorized"})
                return self._send(200, app.metrics.render(app.chunk_count()), "text/plain; version=0.0.4")
            self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/v1/ask":
                return self._send(404, {"error": "not found"})
            ok, token_role = app.authenticate(self.headers.get("Authorization"))
            if not ok:
                return self._send(401, {"error": "unauthorized"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._send(400, {"error": "bad Content-Length"})
            if length <= 0 or length > app.settings.max_body_bytes:
                return self._send(413 if length > 0 else 400, {"error": f"body must be 1..{app.settings.max_body_bytes} bytes"})
            try:
                body = json.loads(self.rfile.read(length))
                question = body["question"]
                if not isinstance(question, str):
                    raise TypeError
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": 'body must be JSON like {"question": "..."}'})
            role = token_role or app.settings.default_role
            if app.settings.allow_role_in_body and isinstance(body.get("role"), str):
                role = body["role"].strip().lower()
            if not role or set(role) - ROLE_OK:
                return self._send(400, {"error": "invalid role"})
            if not app.slots.acquire(blocking=False):
                return self._send(429, {"error": "too many concurrent requests"})
            try:
                res = app.agent.ask(question, role)
            finally:
                app.slots.release()
            app.metrics.record(res.status, res.metrics)
            log.info("ask", extra={"fields": {"status": res.status, "reason": res.reason, "role": role, **{k: res.metrics[k] for k in ("tokens_in", "tokens_out", "cost_usd", "wall_s")}}})
            self._send(200, res.as_dict(include_trace=bool(body.get("trace"))))

    return Handler


def make_server(settings: Settings, agent: Agent, store) -> tuple[ThreadingHTTPServer, App]:
    app = App(agent, settings, store.count)
    srv = ThreadingHTTPServer((settings.host, settings.port), make_handler(app))
    srv.daemon_threads = False  # non-daemon: server_close() waits for in-flight requests (graceful drain)
    return srv, app


def main() -> int:
    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logging.basicConfig(level=settings.log_level, handlers=[handler], force=True)
    try:
        agent, store, report = build(settings, AgentConfig())
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    log.info("ingested", extra={"fields": report.__dict__})
    srv, _app = make_server(settings, agent, store)
    stopping = threading.Event()

    def stop(_sig, _frm):
        if not stopping.is_set():
            stopping.set()
            log.info("shutdown: draining in-flight requests")
            threading.Thread(target=srv.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("listening", extra={"fields": {"host": settings.host, "port": settings.port, "llm": settings.llm}})
    t0 = time.time()
    srv.serve_forever()
    srv.server_close()  # joins non-daemon request threads
    log.info("stopped", extra={"fields": {"uptime_s": round(time.time() - t0, 1)}})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
