"""Helpers to benchmark inference runtime for various engines."""

from __future__ import annotations

import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class BenchmarkResult:
    label: str
    latencies_ms: List[float]

    @property
    def median_ms(self) -> float:
        return statistics.median(self.latencies_ms)

    @property
    def p95_ms(self) -> float:
        if len(self.latencies_ms) < 2:
            return self.latencies_ms[0] if self.latencies_ms else 0.0
        return statistics.quantiles(self.latencies_ms, n=20)[-1]

    @property
    def throughput_tps(self) -> float:
        avg_latency = statistics.mean(self.latencies_ms)
        return 1000.0 / avg_latency if avg_latency else 0.0


@contextmanager
def timed_section(latencies: List[float]) -> Iterable[None]:
    start = time.perf_counter()
    yield
    elapsed = (time.perf_counter() - start) * 1000.0
    latencies.append(elapsed)
