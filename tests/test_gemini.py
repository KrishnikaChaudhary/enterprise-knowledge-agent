import pytest

from kagent.errors import ConfigError
from kagent.gemini import GeminiAuth, GeminiEmbedder, GeminiLLM, Tier

TIERS = {"small": Tier("small-model", 0.1, 0.4), "large": Tier("large-model", 1.0, 4.0)}


def test_aistudio_generation_streams_with_real_usage_and_key_header(fake_gemini):
    llm = GeminiLLM(GeminiAuth("aistudio", api_key="k-123", base_url=fake_gemini.url), TIERS)
    res = llm.complete("answer", "hello", "large")
    path, headers, body = fake_gemini.requests[0]
    assert path == "/models/large-model:streamGenerateContent?alt=sse"
    low = {k.lower(): v for k, v in headers.items()}
    assert low["x-goog-api-key"] == "k-123" and "authorization" not in low
    assert body["generationConfig"]["temperature"] == 0 and body["contents"][0]["parts"][0]["text"] == "hello"
    assert res.text == "Employees get 24 days [1]."
    assert (res.tokens_in, res.tokens_out) == (321, 9)
    assert res.cost_usd == pytest.approx((321 * 1.0 + 9 * 4.0) / 1e6)
    assert res.ttft_s > 0 and res.latency_s >= res.ttft_s


def test_vertex_uses_bearer_token_and_project_path(fake_gemini):
    auth = GeminiAuth("vertex", project="proj-1", location="asia-south1", base_url=fake_gemini.url, token_fn=lambda: ("tok-xyz", 3600))
    GeminiLLM(auth, TIERS).complete("answer", "hi", "small")
    path, headers, _ = fake_gemini.requests[0]
    assert path.startswith("/projects/proj-1/locations/asia-south1/publishers/google/models/small-model:streamGenerateContent")
    assert {k.lower(): v for k, v in headers.items()}["authorization"] == "Bearer tok-xyz"


def test_vertex_default_base_url_is_regional():
    a = GeminiAuth("vertex", project="p", location="asia-south1", token_fn=lambda: ("t", 100))
    assert a.base == "https://asia-south1-aiplatform.googleapis.com/v1"
    assert GeminiAuth("vertex", project="p", location="global", token_fn=lambda: ("t", 100)).base == "https://aiplatform.googleapis.com/v1"


def test_token_is_cached_until_near_expiry():
    calls = []
    auth = GeminiAuth("vertex", project="p", token_fn=lambda: (calls.append(1) or "t", 3600))
    auth.headers(); auth.headers()
    assert len(calls) == 1


def test_retries_on_429_then_succeeds(fake_gemini):
    fake_gemini.fail_first = 2
    llm = GeminiLLM(GeminiAuth("aistudio", api_key="k", base_url=fake_gemini.url), TIERS, backoff_s=0.01)
    assert llm.complete("answer", "hi", "small").text
    assert len(fake_gemini.requests) == 3


def test_gives_up_after_retries(fake_gemini):
    fake_gemini.fail_first = 99
    llm = GeminiLLM(GeminiAuth("aistudio", api_key="k", base_url=fake_gemini.url), TIERS, max_retries=1, backoff_s=0.01)
    with pytest.raises(RuntimeError, match="failed after 2 attempts"):
        llm.complete("answer", "hi", "small")


def test_embedder_aistudio_batches_and_normalises(fake_gemini):
    emb = GeminiEmbedder(GeminiAuth("aistudio", api_key="k", base_url=fake_gemini.url), "embed-model", dim=8, batch=2)
    vecs = emb.embed(["a", "b", "c"], "query")
    assert len(vecs) == 3 and all(len(v) == 8 for v in vecs)
    assert len(fake_gemini.requests) == 2  # batch of 2 + batch of 1
    req = fake_gemini.requests[0][2]["requests"][0]
    assert req["taskType"] == "RETRIEVAL_QUERY" and req["outputDimensionality"] == 8 and req["model"] == "models/embed-model"
    assert emb.name == "gemini:embed-model:8"


def test_embedder_vertex_predict_shape(fake_gemini):
    auth = GeminiAuth("vertex", project="p", base_url=fake_gemini.url, token_fn=lambda: ("t", 3600))
    vecs = GeminiEmbedder(auth, "embed-model", dim=8).embed(["a"], "document")
    path, _h, body = fake_gemini.requests[0]
    assert path.endswith("/models/embed-model:predict") and body["instances"][0]["task_type"] == "RETRIEVAL_DOCUMENT"
    assert vecs[0][1] == 1.0


def test_embedder_detects_dimension_mismatch(fake_gemini):
    with pytest.raises(RuntimeError, match="size mismatch"):
        GeminiEmbedder(GeminiAuth("aistudio", api_key="k", base_url=fake_gemini.url), "m", dim=16).embed(["a"])


def test_config_errors_are_explicit(monkeypatch):
    with pytest.raises(ConfigError):
        GeminiAuth("aistudio")
    with pytest.raises(ConfigError):
        GeminiAuth("vertex")
    with pytest.raises(ConfigError):
        GeminiAuth("nope")
    for k in ("KB_MODEL_SMALL", "KB_MODEL_LARGE", "KB_PRICE_SMALL", "KB_PRICE_LARGE", "KB_EMBED_MODEL"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(ConfigError, match="KB_MODEL_SMALL"):
        GeminiLLM.from_env()
    monkeypatch.setenv("KB_MODEL_SMALL", "s"); monkeypatch.setenv("KB_MODEL_LARGE", "l")
    with pytest.raises(ConfigError, match="KB_PRICE_SMALL"):
        GeminiLLM.from_env()
    with pytest.raises(ConfigError, match="KB_EMBED_MODEL"):
        GeminiEmbedder.from_env()


def test_full_agent_runs_on_the_gemini_client(fake_gemini, store):
    """The whole pipeline over real HTTP/SSE: retrieval uses the hash store, generation goes through the fake Gemini."""
    from kagent.agent import Agent, AgentConfig
    from kagent.embeddings import HashEmbedder
    from kagent.retrieval import Retriever
    llm = GeminiLLM(GeminiAuth("aistudio", api_key="k", base_url=fake_gemini.url), TIERS)
    r = Agent(Retriever(store, HashEmbedder()), llm, AgentConfig(route=False, verify=False)).ask("How many days of annual leave do full-time employees get?")
    assert r.status == "answered" and "[1]" in r.answer and r.metrics["tokens_in"] == 321
