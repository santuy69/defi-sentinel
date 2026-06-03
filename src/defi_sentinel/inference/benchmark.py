"""GPU inference benchmark suite."""

from __future__ import annotations

import time
import logging
from typing import Any

logger = logging.getLogger(__name__)


def run_benchmark(
    model: str = "TheBloke/Llama-2-7B-Chat-GPTQ",
    backend: str = "auto",
    quantization: str = "gptq-4bit",
    iterations: int = 50,
    prompt: str = "Analyze the current DeFi landscape on Ethereum. What are the top yield opportunities?",
) -> dict[str, Any]:
    """Run inference benchmark and return metrics.

    Args:
        model: HuggingFace model name or local path.
        backend: cuda, rocm, cpu, or auto.
        quantization: fp16, gptq-4bit, awq-4bit, gguf-q4.
        iterations: Number of inference runs.
        prompt: Test prompt for benchmarking.

    Returns:
        Dict with benchmark metrics.
    """
    from defi_sentinel.inference.engine import LLMEngine

    engine = LLMEngine(model=model, backend=backend, quantization=quantization)
    engine.load()

    messages = [{"role": "user", "content": prompt}]

    latencies = []
    tokens_per_sec_list = []
    total_tokens = 0

    logger.info("Running %d benchmark iterations...", iterations)

    # Warmup
    for _ in range(3):
        engine.chat(messages, max_new_tokens=64)

    # Actual benchmark
    for i in range(iterations):
        result = engine.chat(messages, max_new_tokens=256)
        latencies.append(result["latency_ms"])
        tokens_per_sec_list.append(result["tokens_per_sec"])
        total_tokens += result["tokens"]

        if (i + 1) % 10 == 0:
            logger.info("  Completed %d/%d iterations", i + 1, iterations)

    engine.unload()

    avg_latency = sum(latencies) / len(latencies)
    avg_tps = sum(tokens_per_sec_list) / len(tokens_per_sec_list)
    p50 = sorted(latencies)[len(latencies) // 2]
    p95 = sorted(latencies)[int(len(latencies) * 0.95)]
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]

    # Get VRAM usage
    vram_used = _get_vram_usage()

    return {
        "model": model,
        "backend": engine.backend,
        "quantization": quantization,
        "iterations": iterations,
        "avg_latency_ms": round(avg_latency, 1),
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "p99_latency_ms": round(p99, 1),
        "avg_tokens_per_sec": round(avg_tps, 1),
        "total_tokens": total_tokens,
        "vram_used_mb": vram_used,
    }


def _get_vram_usage() -> int:
    """Get current VRAM usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() // (1024 * 1024)
    except ImportError:
        pass
    return 0


def compare_backends(
    model: str = "TheBloke/Llama-2-7B-Chat-GPTQ",
    iterations: int = 20,
) -> list[dict[str, Any]]:
    """Run benchmark across all available backends for comparison."""
    results = []

    for backend in ["cuda", "rocm", "cpu"]:
        try:
            result = run_benchmark(
                model=model,
                backend=backend,
                iterations=iterations,
            )
            results.append(result)
        except Exception as e:
            logger.warning("Backend %s failed: %s", backend, e)
            results.append({"backend": backend, "error": str(e)})

    return results
