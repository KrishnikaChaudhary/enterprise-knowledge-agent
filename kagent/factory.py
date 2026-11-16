"""Wires settings -> store, embedder, LLM, retriever, agent."""
from __future__ import annotations

from .agent import Agent, AgentConfig
from .embeddings import Embedder, HashEmbedder
from .gemini import GeminiEmbedder, GeminiLLM
from .ingest import IngestReport, ingest
from .llm import LLMClient, MockLLM
from .retrieval import Retriever
from .settings import Settings
from .store import SqliteVectorStore


def make_embedder(kind: str) -> Embedder:
    return GeminiEmbedder.from_env() if kind == "gemini" else HashEmbedder()


def make_llm(kind: str) -> LLMClient:
    return GeminiLLM.from_env() if kind == "gemini" else MockLLM()


def build(settings: Settings, cfg: AgentConfig | None = None, *, db_path: str | None = None,
          embed_kind: str | None = None) -> tuple[Agent, SqliteVectorStore, IngestReport]:
    store = SqliteVectorStore(db_path or settings.db_path)
    embedder = make_embedder(embed_kind or settings.llm)
    report = ingest(settings.corpus_dir, store, embedder)
    agent = Agent(Retriever(store, embedder), make_llm(settings.llm), cfg, settings.max_cost_usd, settings.max_llm_calls, settings.max_total_tokens)
    return agent, store, report
