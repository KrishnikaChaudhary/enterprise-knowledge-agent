import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import replace

import pytest

from kagent.agent import Agent, AgentConfig
from kagent.errors import ConfigError
from kagent.llm import LLMResult, MockLLM
from kagent.retrieval import Retriever
from kagent.server import make_server
from kagent.settings import Settings

from conftest import SpyLLM

Q = "How many days of annual leave do full-time employees get?"


def start(store, embedder, llm=None, **overrides):
    s = replace(Settings(), port=0, **overrides)
    agent = Agent(Retriever(store, embedder), llm or MockLLM(), AgentConfig())
    srv, app = make_server(s, agent, store)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def call(url, path, body=None, token=None, method=None, raw=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url + path, data, headers, method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


@pytest.fixture
def srv(store, embedder):
    s, url = start(store, embedder, token_roles=(("tok-emp", "employee"), ("tok-hr", "hr")))
    yield url
    s.shutdown(); s.server_close()


def test_health_ready_and_metrics_auth(srv):
    assert call(srv, "/healthz")[0] == 200
    code, body = call(srv, "/readyz")
    assert code == 200 and json.loads(body)["chunks"] > 30
    assert call(srv, "/metrics")[0] == 401
    assert "kagent_indexed_chunks" in call(srv, "/metrics", token="tok-emp")[1]


def test_auth_required_and_role_comes_from_token(srv):
    assert call(srv, "/v1/ask", {"question": Q})[0] == 401
    assert call(srv, "/v1/ask", {"question": Q}, token="wrong")[0] == 401
    code, body = call(srv, "/v1/ask", {"question": Q}, token="tok-emp")
    assert code == 200 and json.loads(body)["status"] == "answered"
    sal = {"question": "What is the salary band for level L4?"}
    assert json.loads(call(srv, "/v1/ask", sal, token="tok-emp")[1])["status"] == "refused"
    assert "$90,000" in json.loads(call(srv, "/v1/ask", sal, token="tok-hr")[1])["answer"]


def test_role_in_body_cannot_escalate_privileges(srv):
    body = {"question": "What is the salary band for level L4?", "role": "hr"}
    assert json.loads(call(srv, "/v1/ask", body, token="tok-emp")[1])["status"] == "refused"


def test_validation_and_limits(srv):
    assert call(srv, "/v1/ask", raw=b"{not json", token="tok-emp")[0] == 400
    assert call(srv, "/v1/ask", {"q": "x"}, token="tok-emp")[0] == 400
    assert call(srv, "/v1/ask", {"question": 5}, token="tok-emp")[0] == 400
    assert call(srv, "/v1/ask", raw=b"x" * 20000, token="tok-emp")[0] == 413
    assert call(srv, "/nope", token="tok-emp")[0] == 404
    assert call(srv, "/v1/ask", {"question": "ok", "trace": True}, token="tok-emp")[0] == 200


def test_trace_is_opt_in(srv):
    assert "trace" not in json.loads(call(srv, "/v1/ask", {"question": Q}, token="tok-emp")[1])
    assert json.loads(call(srv, "/v1/ask", {"question": Q, "trace": True}, token="tok-emp")[1])["trace"]["spans"]


def test_metrics_count_requests_tokens_and_cost(srv):
    call(srv, "/v1/ask", {"question": Q}, token="tok-emp")
    text = call(srv, "/metrics", token="tok-emp")[1]
    assert 'kagent_requests_total{status="answered"}' in text and "kagent_cost_usd_total" in text


def test_no_token_map_means_platform_auth_and_fixed_default_role(store, embedder):
    s, url = start(store, embedder, default_role="employee")
    try:
        assert json.loads(call(url, "/v1/ask", {"question": "What is the salary band for level L4?", "role": "hr"})[1])["status"] == "refused"
    finally:
        s.shutdown(); s.server_close()


def test_body_role_only_when_explicitly_allowed_on_loopback(store, embedder):
    s, url = start(store, embedder, allow_role_in_body=True)
    try:
        assert "$90,000" in json.loads(call(url, "/v1/ask", {"question": "What is the salary band for level L4?", "role": "hr"})[1])["answer"]
        assert call(url, "/v1/ask", {"question": "hi", "role": "HR; DROP"})[0] == 400
    finally:
        s.shutdown(); s.server_close()


class Slow:
    def __init__(self, gate): self.gate, self.inner = gate, MockLLM()
    def complete(self, role, prompt, model):
        self.gate.wait(5)
        return self.inner.complete(role, prompt, model)


def test_concurrency_cap_returns_429(store, embedder):
    gate = threading.Event()
    s, url = start(store, embedder, Slow(gate), max_concurrent=1)
    out = []
    t = threading.Thread(target=lambda: out.append(call(url, "/v1/ask", {"question": Q})))
    t.start()
    time.sleep(0.3)
    try:
        assert call(url, "/v1/ask", {"question": Q})[0] == 429
    finally:
        gate.set(); t.join(5); s.shutdown(); s.server_close()
    assert out[0][0] == 200


def test_shutdown_drains_in_flight_requests(store, embedder):
    gate = threading.Event()
    s, url = start(store, embedder, Slow(gate))
    out = []
    t = threading.Thread(target=lambda: out.append(call(url, "/v1/ask", {"question": Q})))
    t.start()
    time.sleep(0.3)
    stopper = threading.Thread(target=lambda: (s.shutdown(), s.server_close()))
    stopper.start()
    time.sleep(0.3)
    assert stopper.is_alive()  # server_close is waiting for the in-flight request
    gate.set()
    t.join(5); stopper.join(5)
    assert out and out[0][0] == 200 and json.loads(out[0][1])["status"] == "answered"


def test_llm_outage_is_a_200_with_refused_status_not_a_500(store, embedder):
    s, url = start(store, embedder, SpyLLM(raises={"router": TimeoutError("slow")}))
    try:
        code, body = call(url, "/v1/ask", {"question": Q})
        assert code == 200 and json.loads(body)["reason"].startswith("llm_error")
    finally:
        s.shutdown(); s.server_close()


# ---------------- settings ----------------
def test_settings_defaults_and_cloud_run_binding():
    s = Settings.from_env({})
    assert (s.host, s.port, s.llm) == ("127.0.0.1", 8080, "mock")
    cr = Settings.from_env({"K_SERVICE": "svc", "PORT": "9090"})
    assert cr.host == "0.0.0.0" and cr.port == 9090  # Cloud Run: IAM authenticates, so no token map is required


@pytest.mark.parametrize("env", [
    {"KB_HOST": "0.0.0.0"},                          # public bind without tokens outside Cloud Run
    {"KB_LLM": "gpt"},
    {"PORT": "abc"},
    {"KB_MAX_COST_USD": "0"},
    {"KB_TOKENS": "justatoken"},
    {"KB_HOST": "0.0.0.0", "KB_TOKENS": "a=hr", "KB_ALLOW_ROLE_IN_BODY": "1"},
])
def test_settings_reject_bad_config(env):
    with pytest.raises(ConfigError):
        Settings.from_env(env)


def test_token_map_parsing():
    s = Settings.from_env({"KB_HOST": "0.0.0.0", "KB_TOKENS": "a=hr, b=Employee"})
    assert s.token_roles == (("a", "hr"), ("b", "employee"))
