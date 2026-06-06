from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheConfig:
    """Static KV-cache shape for one decoder-only model."""

    layers: int = 32
    kv_heads: int = 32
    head_dim: int = 128
    bytes_per_element: int = 2

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.bytes_per_element

    def request_bytes(self, prompt_tokens: int, output_tokens: int) -> int:
        return (prompt_tokens + output_tokens) * self.bytes_per_token


def bytes_to_mib(value: int | float) -> float:
    return value / (1024 * 1024)

