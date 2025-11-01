"""Simple engine wrapper used by scripts to execute inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class Backend(Protocol):
    def warmup(self, prompt: Iterable[int]) -> None: ...
    def generate(self, prompt: list[int], max_tokens: int) -> list[int]: ...


@dataclass
class InferenceEngine:
    backend: Backend

    def run(self, prompt: list[int], *, max_tokens: int = 16) -> list[int]:
        self.backend.warmup(prompt)
        return self.backend.generate(prompt, max_tokens=max_tokens)
