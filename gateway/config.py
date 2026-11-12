"""Gateway configuration (YAML). Secrets are referenced by environment variable name, never inlined."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class Client(BaseModel):
    name: str
    key_env: str = Field(description="Environment variable holding this client's bearer token")
    rpm: int = Field(60, ge=1, description="Requests per minute")

    @property
    def key(self) -> str:
        return os.getenv(self.key_env, "")


class Upstream(BaseModel):
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: float = Field(60, gt=0)

    @property
    def api_key(self) -> str:
        return os.getenv(self.api_key_env, "") if self.api_key_env else ""


class Target(BaseModel):
    upstream: str
    model: str


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_seconds: float = Field(3600, gt=0)
    max_entries: int = Field(5000, ge=1)
    semantic: bool = True
    semantic_threshold: float = Field(0.92, gt=0, le=1)


class BreakerConfig(BaseModel):
    failure_threshold: int = Field(3, ge=1)
    cooldown_seconds: float = Field(30, gt=0)


class Config(BaseModel):
    clients: list[Client] = Field(min_length=1)
    upstreams: dict[str, Upstream] = Field(min_length=1)
    routes: dict[str, list[Target]] = Field(min_length=1, description="model alias -> targets in fallback order")
    cache: CacheConfig = CacheConfig()
    breaker: BreakerConfig = BreakerConfig()
    redact_pii: bool = Field(False, description="Mask emails, phone numbers and card numbers before sending upstream")

    @model_validator(mode="after")
    def _references_resolve(self) -> Config:
        for alias, targets in self.routes.items():
            if not targets:
                raise ValueError(f"route {alias!r} has no targets")
            for target in targets:
                if target.upstream not in self.upstreams:
                    raise ValueError(f"route {alias!r} points at unknown upstream {target.upstream!r}")
        names = [c.name for c in self.clients]
        if len(names) != len(set(names)):
            raise ValueError("client names must be unique")
        return self


def load_config(path: str | Path) -> Config:
    return Config.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
