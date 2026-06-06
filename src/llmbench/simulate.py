from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from llmbench.kv_cache import KVCacheConfig, bytes_to_mib
from llmbench.workload import RequestSpec, Workload


@dataclass(frozen=True)
class LatencyModel:
    scheduler_overhead_ms: float = 0.20
    tokenize_ms_per_prompt_token: float = 0.004
    prefill_ms_per_prompt_token: float = 0.030
    decode_ms_per_output_token: float = 2.250
    stream_ms_per_output_token: float = 0.010


@dataclass(frozen=True)
class RequestTrace:
    request_id: str
    arrival_ms: float
    start_ms: float
    end_ms: float
    queue_wait_ms: float
    scheduler_ms: float
    tokenize_ms: float
    prefill_ms: float
    decode_ms: float
    stream_ms: float
    prompt_tokens: int
    output_tokens: int
    kv_cache_bytes: int

    @property
    def latency_ms(self) -> float:
        return self.end_ms - self.arrival_ms

    @property
    def ttft_ms(self) -> float:
        first_decode_ms = min(self.decode_ms, self.decode_ms / max(1, self.output_tokens))
        return (
            self.queue_wait_ms
            + self.scheduler_ms
            + self.tokenize_ms
            + self.prefill_ms
            + first_decode_ms
        )

    @property
    def tpot_ms(self) -> float:
        if self.output_tokens <= 1:
            return 0.0
        return self.decode_ms / self.output_tokens

    @property
    def kv_cache_mib(self) -> float:
        return bytes_to_mib(self.kv_cache_bytes)


def simulate_fifo(
    workload: Workload,
    model: LatencyModel | None = None,
    kv_cache: KVCacheConfig | None = None,
) -> list[RequestTrace]:
    latency_model = model or LatencyModel()
    kv_cache_config = kv_cache or KVCacheConfig()
    traces: list[RequestTrace] = []
    worker_available_ms = 0.0

    for request in workload.requests:
        trace = _simulate_request(request, worker_available_ms, latency_model, kv_cache_config)
        traces.append(trace)
        worker_available_ms = trace.end_ms

    return traces


def summarize_traces(traces: list[RequestTrace]) -> dict[str, float]:
    if not traces:
        return {}

    latencies = sorted(trace.latency_ms for trace in traces)
    queue_waits = sorted(trace.queue_wait_ms for trace in traces)
    total_output_tokens = sum(trace.output_tokens for trace in traces)
    kv_cache_sizes = sorted(trace.kv_cache_mib for trace in traces)
    start_ms = min(trace.arrival_ms for trace in traces)
    end_ms = max(trace.end_ms for trace in traces)
    elapsed_s = max((end_ms - start_ms) / 1000.0, 1e-9)

    return {
        "requests": float(len(traces)),
        "output_tokens": float(total_output_tokens),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "p95_queue_wait_ms": _percentile(queue_waits, 95),
        "max_request_kv_cache_mib": max(kv_cache_sizes),
        "total_request_kv_cache_mib": sum(kv_cache_sizes),
        "requests_per_second": len(traces) / elapsed_s,
        "output_tokens_per_second": total_output_tokens / elapsed_s,
    }


def _simulate_request(
    request: RequestSpec,
    worker_available_ms: float,
    model: LatencyModel,
    kv_cache: KVCacheConfig,
) -> RequestTrace:
    start_ms = max(request.arrival_ms, worker_available_ms)
    queue_wait_ms = start_ms - request.arrival_ms
    scheduler_ms = model.scheduler_overhead_ms
    tokenize_ms = request.prompt_tokens * model.tokenize_ms_per_prompt_token
    prefill_ms = request.prompt_tokens * model.prefill_ms_per_prompt_token
    decode_ms = request.output_tokens * model.decode_ms_per_output_token
    stream_ms = request.output_tokens * model.stream_ms_per_output_token
    kv_cache_bytes = kv_cache.request_bytes(request.prompt_tokens, request.output_tokens)
    end_ms = start_ms + scheduler_ms + tokenize_ms + prefill_ms + decode_ms + stream_ms

    return RequestTrace(
        request_id=request.id,
        arrival_ms=request.arrival_ms,
        start_ms=start_ms,
        end_ms=end_ms,
        queue_wait_ms=queue_wait_ms,
        scheduler_ms=scheduler_ms,
        tokenize_ms=tokenize_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        stream_ms=stream_ms,
        prompt_tokens=request.prompt_tokens,
        output_tokens=request.output_tokens,
        kv_cache_bytes=kv_cache_bytes,
    )


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    rank = ceil((percentile / 100) * len(values))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]
