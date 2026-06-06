from __future__ import annotations

import heapq
from dataclasses import dataclass
from math import ceil

from llmbench.kv_cache import (
    KVCacheConfig,
    KVCachePoint,
    bytes_to_mib,
    summarize_kv_cache_timeline,
)
from llmbench.workload import RequestSpec, Workload

SCHEDULING_POLICIES = (
    "fifo",
    "shortest-prefill",
    "shortest-cache",
    "shortest-service",
    "deadline",
)


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
    deadline_ms: float | None
    kv_cache_bytes_per_token: int
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

    @property
    def absolute_deadline_ms(self) -> float | None:
        if self.deadline_ms is None:
            return None
        return self.arrival_ms + self.deadline_ms

    @property
    def deadline_lateness_ms(self) -> float | None:
        absolute_deadline = self.absolute_deadline_ms
        if absolute_deadline is None:
            return None
        return max(0.0, self.end_ms - absolute_deadline)

    @property
    def scheduler_end_ms(self) -> float:
        return self.start_ms + self.scheduler_ms

    @property
    def tokenize_end_ms(self) -> float:
        return self.scheduler_end_ms + self.tokenize_ms

    @property
    def prefill_start_ms(self) -> float:
        return self.tokenize_end_ms

    @property
    def prefill_end_ms(self) -> float:
        return self.prefill_start_ms + self.prefill_ms

    @property
    def decode_start_ms(self) -> float:
        return self.prefill_end_ms

    @property
    def decode_end_ms(self) -> float:
        return self.decode_start_ms + self.decode_ms


def simulate_fifo(
    workload: Workload,
    model: LatencyModel | None = None,
    kv_cache: KVCacheConfig | None = None,
    max_concurrent_requests: int = 1,
) -> list[RequestTrace]:
    return simulate_scheduler(
        workload,
        model=model,
        kv_cache=kv_cache,
        max_concurrent_requests=max_concurrent_requests,
        scheduling_policy="fifo",
    )


def simulate_scheduler(
    workload: Workload,
    model: LatencyModel | None = None,
    kv_cache: KVCacheConfig | None = None,
    max_concurrent_requests: int = 1,
    scheduling_policy: str = "fifo",
) -> list[RequestTrace]:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be positive")
    if scheduling_policy not in SCHEDULING_POLICIES:
        policies = ", ".join(SCHEDULING_POLICIES)
        raise ValueError(f"unknown scheduling_policy {scheduling_policy!r}; expected one of: {policies}")

    latency_model = model or LatencyModel()
    kv_cache_config = kv_cache or KVCacheConfig()
    requests = list(workload.requests)
    if not requests:
        return []

    traces: list[RequestTrace] = []
    running: list[float] = []
    waiting: list[tuple[int, RequestSpec]] = []
    next_request_index = 0
    now_ms = requests[0].arrival_ms

    while next_request_index < len(requests) or waiting or running:
        while next_request_index < len(requests) and requests[next_request_index].arrival_ms <= now_ms:
            waiting.append((next_request_index, requests[next_request_index]))
            next_request_index += 1

        while running and running[0] <= now_ms:
            heapq.heappop(running)

        while waiting and len(running) < max_concurrent_requests:
            _, request = _pop_next_request(
                waiting,
                scheduling_policy,
                latency_model,
                kv_cache_config,
            )
            trace = _simulate_request(request, now_ms, latency_model, kv_cache_config)
            traces.append(trace)
            heapq.heappush(running, trace.end_ms)

        next_arrival_ms = (
            requests[next_request_index].arrival_ms
            if next_request_index < len(requests)
            else None
        )
        next_completion_ms = running[0] if running else None
        next_time_ms = _next_event_time(
            now_ms,
            next_arrival_ms,
            next_completion_ms,
            waiting=bool(waiting),
            at_capacity=len(running) >= max_concurrent_requests,
        )
        if next_time_ms is None:
            break
        now_ms = next_time_ms

    return traces


def summarize_traces(traces: list[RequestTrace]) -> dict[str, float]:
    if not traces:
        return {}

    latencies = sorted(trace.latency_ms for trace in traces)
    queue_waits = sorted(trace.queue_wait_ms for trace in traces)
    deadline_lateness = sorted(
        lateness
        for trace in traces
        if (lateness := trace.deadline_lateness_ms) is not None
    )
    total_output_tokens = sum(trace.output_tokens for trace in traces)
    kv_cache_sizes = sorted(trace.kv_cache_mib for trace in traces)
    timeline = build_kv_cache_timeline(traces)
    timeline_summary = summarize_kv_cache_timeline(timeline)
    start_ms = min(trace.arrival_ms for trace in traces)
    end_ms = max(trace.end_ms for trace in traces)
    elapsed_s = max((end_ms - start_ms) / 1000.0, 1e-9)

    summary = {
        "requests": float(len(traces)),
        "output_tokens": float(total_output_tokens),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "p95_queue_wait_ms": _percentile(queue_waits, 95),
        "max_request_kv_cache_mib": max(kv_cache_sizes),
        "total_request_kv_cache_mib": sum(kv_cache_sizes),
        **timeline_summary,
        "requests_per_second": len(traces) / elapsed_s,
        "output_tokens_per_second": total_output_tokens / elapsed_s,
    }

    if deadline_lateness:
        misses = sum(1 for value in deadline_lateness if value > 0)
        summary.update(
            {
                "deadline_tracked_requests": float(len(deadline_lateness)),
                "deadline_miss_rate": misses / len(deadline_lateness),
                "p95_deadline_lateness_ms": _percentile(deadline_lateness, 95),
            }
        )

    return summary


def build_kv_cache_timeline(traces: list[RequestTrace]) -> list[KVCachePoint]:
    """Build an active KV-cache timeline from request traces.

    The model is intentionally simple:
    - prompt KV is allocated at prefill start
    - one output token worth of KV is added at each decode step
    - all request KV is released at request completion
    """

    events: dict[float, list[int]] = {}
    for trace in traces:
        _add_event(
            events,
            trace.prefill_start_ms,
            trace.prompt_tokens * trace.kv_cache_bytes_per_token,
        )

        if trace.output_tokens > 0:
            decode_step_ms = trace.decode_ms / trace.output_tokens
            for index in range(1, trace.output_tokens + 1):
                event_time_ms = trace.decode_start_ms + index * decode_step_ms
                _add_event(events, event_time_ms, trace.kv_cache_bytes_per_token)

        _add_event(events, trace.end_ms, -trace.kv_cache_bytes)

    active_bytes = 0
    points: list[KVCachePoint] = []
    for time_ms in sorted(events):
        deltas = events[time_ms]
        delta_bytes = sum(deltas)
        active_bytes += delta_bytes
        points.append(
            KVCachePoint(
                time_ms=time_ms,
                active_bytes=max(active_bytes, 0),
                delta_bytes=delta_bytes,
                events=len(deltas),
            )
        )

    return points


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
    kv_cache_bytes_per_token = kv_cache.bytes_per_token
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
        deadline_ms=request.deadline_ms,
        kv_cache_bytes_per_token=kv_cache_bytes_per_token,
        kv_cache_bytes=kv_cache_bytes,
    )


def _pop_next_request(
    waiting: list[tuple[int, RequestSpec]],
    scheduling_policy: str,
    model: LatencyModel,
    kv_cache: KVCacheConfig,
) -> tuple[int, RequestSpec]:
    best_index = min(
        range(len(waiting)),
        key=lambda index: _scheduler_key(
            waiting[index][0],
            waiting[index][1],
            scheduling_policy,
            model,
            kv_cache,
        ),
    )
    return waiting.pop(best_index)


def _scheduler_key(
    arrival_order: int,
    request: RequestSpec,
    scheduling_policy: str,
    model: LatencyModel,
    kv_cache: KVCacheConfig,
) -> tuple[float, ...]:
    if scheduling_policy == "fifo":
        return (arrival_order,)
    if scheduling_policy == "shortest-prefill":
        return (request.prompt_tokens, request.arrival_ms, arrival_order)
    if scheduling_policy == "shortest-cache":
        return (
            kv_cache.request_bytes(request.prompt_tokens, request.output_tokens),
            request.arrival_ms,
            arrival_order,
        )
    if scheduling_policy == "shortest-service":
        return (_service_time_ms(request, model), request.arrival_ms, arrival_order)
    if scheduling_policy == "deadline":
        deadline = (
            request.arrival_ms + request.deadline_ms
            if request.deadline_ms is not None
            else float("inf")
        )
        return (deadline, request.arrival_ms, arrival_order)
    raise AssertionError(f"unhandled scheduling policy {scheduling_policy}")


def _service_time_ms(request: RequestSpec, model: LatencyModel) -> float:
    return (
        model.scheduler_overhead_ms
        + request.prompt_tokens * model.tokenize_ms_per_prompt_token
        + request.prompt_tokens * model.prefill_ms_per_prompt_token
        + request.output_tokens * model.decode_ms_per_output_token
        + request.output_tokens * model.stream_ms_per_output_token
    )


def _next_event_time(
    now_ms: float,
    next_arrival_ms: float | None,
    next_completion_ms: float | None,
    *,
    waiting: bool,
    at_capacity: bool,
) -> float | None:
    if waiting and at_capacity:
        return next_completion_ms
    if next_arrival_ms is None:
        return next_completion_ms
    if not at_capacity:
        return max(now_ms, next_arrival_ms)
    if next_completion_ms is None:
        return next_arrival_ms
    return min(next_arrival_ms, next_completion_ms)


def _add_event(events: dict[float, list[int]], time_ms: float, delta_bytes: int) -> None:
    events.setdefault(round(time_ms, 6), []).append(delta_bytes)


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    rank = ceil((percentile / 100) * len(values))
    index = min(max(rank - 1, 0), len(values) - 1)
    return values[index]
