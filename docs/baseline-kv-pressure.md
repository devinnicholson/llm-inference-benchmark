# Baseline KV-Cache Pressure

## Question

For a simple FIFO LLM serving baseline, how much active KV-cache memory does a
mixed workload create over time?

This is the first measurable research question in the repo. It does not try to
optimize scheduling yet. It makes KV-cache pressure visible so later scheduler,
batching, and cache-management experiments have a concrete target.

## Method

Run the mixed workload through the deterministic FIFO simulator:

```bash
python3 scripts/replay_workload.py workloads/week01_mixed_requests.json \
  --model-config configs/models/llama-7b-gqa-fp16.json
```

The simulator uses this active-memory model:

1. Prompt KV is allocated at prefill start.
2. One output token worth of KV is added at each decode step.
3. The request's KV cache is released when the request completes.

The current FIFO baseline processes one request at a time, so active KV cache
does not overlap across requests. That is an intentional starting point. The
timeline implementation also supports overlapping traces, which is used by the
concurrent FIFO baseline later in this document.

## Model Configs

The first model configs are approximate KV-cache shapes:

| Config | Layers | KV heads | Head dim | Bytes/element | Bytes/token | MiB/token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llama-7b-mha-fp16` | 32 | 32 | 128 | 2 | 524288 | 0.5000 |
| `llama-7b-gqa-fp16` | 32 | 8 | 128 | 2 | 131072 | 0.1250 |
| `llama-13b-mha-fp16` | 40 | 40 | 128 | 2 | 819200 | 0.7812 |
| `llama-7b-gqa-int8-kv` | 32 | 8 | 128 | 1 | 65536 | 0.0625 |

The point is not to claim exact model identities. The point is to compare how
MHA, GQA, larger depth/head counts, and lower-precision KV settings change
memory pressure.

## Initial Observation

For `week01_mixed_requests`, request KV-cache footprint is driven by total
sequence length:

| Request | Prompt tokens | Output tokens | Total tokens | KV MiB, 7B MHA FP16 | KV MiB, 7B GQA FP16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `chat-short-001` | 128 | 64 | 192 | 96.0 | 24.0 |
| `coding-long-001` | 1800 | 320 | 2120 | 1060.0 | 265.0 |
| `rag-medium-001` | 900 | 160 | 1060 | 530.0 | 132.5 |
| `chat-short-002` | 96 | 48 | 144 | 72.0 | 18.0 |
| `summary-batch-001` | 2400 | 220 | 2620 | 1310.0 | 327.5 |

Even before batching, the long-context requests dominate peak active KV memory.
With a 7B MHA-style FP16 cache, the largest request uses about 1.31 GiB of KV
cache. With a 7B GQA-style FP16 cache, the same request uses about 327.5 MiB.

## Current Replay Result

Using `llama-7b-gqa-fp16`, the baseline summary is:

```text
p95_latency_ms               1997.136
p95_queue_wait_ms            1418.136
max_request_kv_cache_mib      327.500
total_request_kv_cache_mib    767.000
peak_active_kv_cache_mib      327.500
p95_active_kv_cache_mib       322.000
memory_pressure_duration_ms   626.400
```

`memory_pressure_duration_ms` is currently defined as the duration where active
KV cache is at or above 80 percent of the run's peak active KV cache. Once we add
GPU capacity configs, this should become capacity-relative instead of
peak-relative.

## Interpretation

The baseline says three useful things:

1. Total tokens are the first-order KV-cache driver.
2. GQA/MQA-style KV head reduction can change the memory profile by multiples.
3. FIFO queue wait is already large for bursty mixed workloads, but this serial
   baseline cannot yet expose cache contention between concurrent requests.

The third point is the next research opening. We need overlapping active-memory
timelines before scheduler claims become meaningful.

## Concurrency Baseline

The simulator now supports FIFO with configurable concurrent request slots:

```bash
python3 scripts/replay_workload.py workloads/generated/mixed_bursty_32_seed568.json \
  --model-config configs/models/llama-7b-gqa-fp16.json \
  --max-concurrent-requests 4
```

On `mixed_bursty_32_seed568`, four concurrent FIFO slots reduce tail latency but
increase peak active KV cache:

| Metric | Serial FIFO | 4-slot FIFO |
| --- | ---: | ---: |
| `p95_latency_ms` | 16325.779 | 3972.368 |
| `p99_latency_ms` | 16599.393 | 4025.487 |
| `p95_queue_wait_ms` | 16153.883 | 3118.553 |
| `peak_active_kv_cache_mib` | 577.625 | 1425.375 |
| `p95_active_kv_cache_mib` | 512.125 | 1377.750 |
| `output_tokens_per_second` | 407.220 | 1517.186 |

That is the first concrete tradeoff this repo can show: concurrency improves
queueing and throughput, but it creates overlapping KV-cache residency. This is
where cache-aware scheduling becomes meaningful.

## Next Step

Add a scheduler interface so we can compare FIFO against policies such as:

- shortest-prefill first
- shortest-cache-footprint first
- deadline-aware scheduling
- chunked-prefill scheduling

The main metrics should be p95/p99 latency, TTFT, TPOT, peak active KV memory,
p95 active KV memory, and duration over a GPU-memory pressure threshold.
