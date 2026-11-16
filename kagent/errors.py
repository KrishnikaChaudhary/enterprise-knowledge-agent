"""Exception types. Everything raised on purpose derives from KagentError."""
from __future__ import annotations


class KagentError(Exception):
    """Base class for expected, handled failures."""


class BudgetExceeded(KagentError):
    def __init__(self, what: str, limit: float, used: float):
        super().__init__(f"budget exceeded: {what} used {used:g} > limit {limit:g}")
        self.what, self.limit, self.used = what, limit, used


class IngestError(KagentError):
    """A source file could not be parsed."""


class ConfigError(KagentError):
    """Missing or invalid configuration."""
