# Week 1: Request Lifecycle and Measurement Vocabulary

## Learning Objective

By the end of Week 1, we should be able to explain a single inference request
from arrival to final token, name the major latency components, and define a
workload format that later experiments can reuse.

## Mental Model

An LLM serving request is not just "call model.generate". A useful systems view
breaks the path into stages:

1. **Arrival**: the request reaches the serving system.
2. **Admission**: the system decides whether to accept, queue, reject, or shed.
3. **Queue wait**: the request waits for scheduler capacity.
4. **Tokenization**: text becomes token IDs.
5. **Prefill**: the model processes the prompt and initializes KV cache.
6. **Decode**: the model generates output tokens one step at a time.
7. **Streaming**: tokens are emitted to the client.
8. **Accounting**: metrics, traces, billing counters, and logs are finalized.

This first implementation simulates those stages so we can start with clean
measurement semantics before adding a real model backend.

## Metrics We Care About

- **End-to-end latency**: request completion time minus arrival time.
- **Queue wait**: scheduling start time minus arrival time.
- **TTFT**: time to first token. In a real backend, this usually includes queue
  wait, tokenization, prefill, scheduler overhead, and first decode step.
- **TPOT**: time per output token after the first token.
- **Throughput**: completed requests or generated tokens per second.
- **Tail latency**: p95/p99 latency, often more important than averages.

## Week 1 Build

The code in `src/llmbench` creates a deterministic FIFO baseline:

- `workload.py` loads and validates request specs.
- `kv_cache.py` estimates per-token and per-request KV-cache memory.
- `simulate.py` estimates per-stage timings and active KV-cache timelines.
- `scripts/replay_workload.py` prints traces and summary metrics.
- `tests/test_workload.py` checks validation and metric behavior.

This is not the final serving system. It is the benchmark control plane we will
use to avoid vague results later.

## Deliverable

Commit a working baseline where this command succeeds:

```bash
python3 scripts/replay_workload.py workloads/week01_mixed_requests.json
python3 -m unittest discover -s tests
```

The README should explain what the simulator does, what it does not do, and how
we will replace each simulated stage with real measurements over time.

## KV-Cache Accounting

The first memory model is intentionally simple:

```text
bytes_per_token = 2 * layers * kv_heads * head_dim * bytes_per_element
request_bytes = (prompt_tokens + output_tokens) * bytes_per_token
```

It is enough to make long-context memory pressure visible in every trace. Later
weeks should add paged allocation, fragmentation, and cache-aware scheduling.

## Active Timeline

The baseline now tracks active KV cache over time:

1. prompt KV allocation at prefill start
2. output-token KV growth during decode
3. full release at request completion

The summary reports peak active KV cache, p95 active KV cache, and duration near
the run's peak memory pressure. The first baseline writeup lives in
`docs/baseline-kv-pressure.md`.
