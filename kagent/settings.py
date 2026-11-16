"""12-factor configuration from KB_* environment variables, validated at startup."""
from __future__ import annotations

import os
from dataclasses import dataclass

from .errors import ConfigError


def _int(env, name: str, default: int) -> int:
    raw = env.get(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(env, name: str, default: float) -> float:
    raw = env.get(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _pairs(raw: str) -> tuple[tuple[str, str], ...]:
    out = []
    for item in filter(None, (p.strip() for p in raw.split(","))):
        tok, sep, role = item.partition("=")
        if not sep or not tok or not role:
            raise ConfigError("KB_TOKENS must look like token1=role1,token2=role2")
        out.append((tok, role.strip().lower()))
    return tuple(out)


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080
    token_roles: tuple[tuple[str, str], ...] = ()  # (bearer token, role) pairs from KB_TOKENS="tok1=hr,tok2=employee"
    default_role: str = "employee"  # role when no token map is configured (e.g. behind Cloud Run IAM)
    allow_role_in_body: bool = False  # dev/eval only: lets the request body pick its own role
    corpus_dir: str = "data/corpus"
    db_path: str = "kb.db"
    llm: str = "mock"  # mock | gemini
    max_concurrent: int = 8
    max_body_bytes: int = 16384
    max_cost_usd: float = 0.10
    max_llm_calls: int = 8
    max_total_tokens: int = 30000
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env
        s = cls(
            host=env.get("KB_HOST", "0.0.0.0" if env.get("K_SERVICE") else "127.0.0.1"),
            port=_int(env, "PORT", _int(env, "KB_PORT", 8080)),
            token_roles=_pairs(env.get("KB_TOKENS", "")),
            default_role=env.get("KB_DEFAULT_ROLE", "employee").strip().lower(),
            allow_role_in_body=env.get("KB_ALLOW_ROLE_IN_BODY", "") == "1",
            corpus_dir=env.get("KB_CORPUS_DIR", "data/corpus"),
            db_path=env.get("KB_DB", "kb.db"),
            llm=env.get("KB_LLM", "mock"),
            max_concurrent=_int(env, "KB_MAX_CONCURRENT", 8),
            max_body_bytes=_int(env, "KB_MAX_BODY_BYTES", 16384),
            max_cost_usd=_float(env, "KB_MAX_COST_USD", 0.10),
            max_llm_calls=_int(env, "KB_MAX_LLM_CALLS", 8),
            max_total_tokens=_int(env, "KB_MAX_TOTAL_TOKENS", 30000),
            log_level=env.get("KB_LOG_LEVEL", "INFO").upper(),
        )
        if s.llm not in {"mock", "gemini"}:
            raise ConfigError("KB_LLM must be mock or gemini")
        if not (1 <= s.port <= 65535) or s.max_concurrent < 1 or s.max_cost_usd <= 0:
            raise ConfigError("invalid port / concurrency / cost limit")
        if s.host not in {"127.0.0.1", "localhost", "::1"} and not s.token_roles and not env.get("K_SERVICE"):
            raise ConfigError("KB_TOKENS is required when binding a non-loopback host outside Cloud Run (where IAM authenticates callers)")
        if s.allow_role_in_body and s.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigError("KB_ALLOW_ROLE_IN_BODY=1 is only allowed on a loopback host: callers could grant themselves any role")
        return s
