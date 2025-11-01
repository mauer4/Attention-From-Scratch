"""Utility functions to compare inference engines using benchmark results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

from .benchmark_runtime import BenchmarkResult


@dataclass
class ComparisonRow:
    baseline_label: str
    contender_label: str
    speedup: float
    latency_delta_ms: float


def compare(baseline: BenchmarkResult, contender: BenchmarkResult) -> ComparisonRow:
    speedup = (
        baseline.throughput_tps / contender.throughput_tps
        if contender.throughput_tps
        else 0.0
    )
    latency_delta = contender.median_ms - baseline.median_ms
    return ComparisonRow(
        baseline_label=baseline.label,
        contender_label=contender.label,
        speedup=round(speedup, 3),
        latency_delta_ms=round(latency_delta, 3),
    )


def rank(results: Iterable[BenchmarkResult]) -> List[BenchmarkResult]:
    return sorted(results, key=lambda res: res.median_ms)
