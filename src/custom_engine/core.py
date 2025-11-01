"""Minimal custom engine stub used for integration tests and demos."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List


@dataclass
class CustomEngine:
    """Toy engine implementation returning deterministic token sequences."""

    name: str = "custom_engine_stub"

    def warmup(self, prompt: Iterable[int]) -> None:
        # Warmup just iterates over tokens to mimic minimal pre-processing.
        for _ in prompt:
            pass

    def generate(self, prompt: List[int], max_tokens: int = 16) -> List[int]:
        result = list(prompt)
        result.extend(range(max_tokens))
        return result


def load_engine(**_: Any) -> CustomEngine:
    """Factory for the stub engine so scripts can obtain an instance."""
    return CustomEngine()
