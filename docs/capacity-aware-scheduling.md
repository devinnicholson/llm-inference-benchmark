# Capacity-Aware Scheduling Baseline

## Question

Under a constrained KV-cache budget, can memory-aware scheduling reduce budget
violations while preserving latency and deadline behavior?

This milestone adds a serving capacity model. Model configs still define KV
bytes per token. Capacity configs define the memory available for KV cache after
model weights and runtime reserve.

## Capacity Config

Example:

```json
{
  "name": "tight-1gb-kv",
  "total_memory_mib": 24576,
  "model_weights_mib": 21000,
  "runtime_reserved_mib": 2552,
  "kv_cache_budget_mib": 1024
}
```

The effective KV-cache budget is:

```text
kv_cache_budget_mib = total_memory_mib - model_weights_mib - runtime_reserved_mib
```

The config can also set `kv_cache_budget_mib` explicitly.

## New Metrics

When `--capacity-config` is provided, replay summaries include:

```text
kv_cache_budget_mib
peak_kv_budget_utilization
p95_kv_budget_utilization
kv_budget_pressure_duration_ms
kv_budget_exceeded_duration_ms
```

`kv_budget_pressure_duration_ms` measures time at or above 80 percent of the
KV-cache budget. `kv_budget_exceeded_duration_ms` measures time above the budget.

## Memory-Aware Policy

The new policy is:

```text
memory-aware-deadline
```

It works as follows:

1. Track currently running requests.
2. Estimate active KV cache at the candidate request's prompt-allocation time.
3. Filter waiting requests whose prompt KV allocation fits the current budget.
4. Among requests that fit, choose the earliest deadline.
5. If no waiting request fits, wait for another arrival or completion.

This is intentionally prompt-admission-aware, not full-lifetime-aware. Decode
growth can still exceed the budget, and that residual violation is measured.

## Command

```bash
python3 scripts/replay_workload.py workloads/generated/mixed_bursty_32_seed568.json \
  --model-config configs/models/llama-7b-gqa-fp16.json \
  --capacity-config configs/capacity/tight-1gb-kv.json \
  --max-concurrent-requests 4 \
  --scheduler-policy memory-aware-deadline
```

## Initial Result

Workload: `mixed_bursty_32_seed568`

Model config: `llama-7b-gqa-fp16`

Capacity config: `tight-1gb-kv`

Concurrent request slots: `4`

| Policy | p95 latency | Deadline miss rate | Peak budget util | p95 budget util | Budget exceeded ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fifo` | 3576.177 | 0.438 | 1.121 | 1.022 | 322.343 |
| `shortest-cache` | 3471.038 | 0.219 | 1.365 | 1.270 | 465.071 |
| `shortest-service` | 3692.655 | 0.188 | 1.180 | 1.100 | 257.021 |
| `deadline` | 3736.294 | 0.000 | 1.169 | 1.124 | 625.775 |
| `memory-aware-deadline` | 3736.294 | 0.000 | 1.043 | 0.997 | 197.510 |

## Interpretation

The first useful result is not that memory-aware scheduling solves memory
pressure. It does not. The result is narrower:

1. Plain `deadline` eliminates synthetic deadline misses but spends 625.775 ms
   over the 1 GiB KV-cache budget.
2. `memory-aware-deadline` keeps the deadline miss rate at zero while reducing
   budget-exceeded duration to 197.510 ms.
3. `shortest-cache` is not automatically memory-safe. It improves deadline miss
   rate relative to FIFO, but it has the highest peak budget utilization in this
   run because multiple smaller choices can still overlap with large active
   requests.

This gives the next research step: move from prompt-admission checks to
full-lifetime or chunk-aware KV budgeting.

## Next Step

Add a stricter policy that estimates the candidate request's full decode growth
against the active timeline, or add chunked prefill/decode so large requests do
not reserve and grow KV cache in one uninterrupted run.

