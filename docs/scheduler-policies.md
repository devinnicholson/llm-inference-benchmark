# Scheduler Policy Baseline

## Question

Given limited concurrent request slots, how do simple request-selection policies
change latency, deadline misses, and active KV-cache pressure?

This is not yet a production scheduler. It is the first policy interface for
testing whether cache-aware and deadline-aware choices beat FIFO under the same
workload.

## Policies

| Policy | Selection rule | Why it matters |
| --- | --- | --- |
| `fifo` | Oldest waiting request first. | Baseline with predictable fairness. |
| `shortest-prefill` | Fewest prompt tokens first. | Reduces long-prefill head-of-line blocking. |
| `shortest-cache` | Smallest total KV-cache footprint first. | Prioritizes requests with lower memory residency. |
| `shortest-service` | Lowest estimated service time first. | Classic latency-oriented queueing heuristic. |
| `deadline` | Earliest absolute deadline first. | Directly optimizes synthetic SLO misses. |
| `memory-aware-deadline` | Earliest deadline among requests whose prompt KV fits capacity. | Adds memory-headroom admission control. |

Generated workloads include `deadline_ms` as a relative latency target. The
absolute deadline is:

```text
absolute_deadline_ms = arrival_ms + deadline_ms
```

## Command

```bash
python3 scripts/replay_workload.py workloads/generated/mixed_bursty_32_seed568.json \
  --model-config configs/models/llama-7b-gqa-fp16.json \
  --max-concurrent-requests 4 \
  --scheduler-policy shortest-cache
```

## Initial Result

Workload: `mixed_bursty_32_seed568`

Model config: `llama-7b-gqa-fp16`

Concurrent request slots: `4`

| Policy | p50 latency | p95 latency | p99 latency | Deadline miss rate | Peak active KV MiB | Output tokens/sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fifo` | 1787.439 | 3576.177 | 3943.293 | 0.438 | 1148.250 | 1459.600 |
| `shortest-prefill` | 1187.824 | 3539.976 | 4402.214 | 0.188 | 1529.375 | 1392.323 |
| `shortest-cache` | 1155.284 | 3471.038 | 4416.717 | 0.219 | 1397.250 | 1388.002 |
| `shortest-service` | 1088.724 | 3692.655 | 4202.302 | 0.188 | 1208.375 | 1454.742 |
| `deadline` | 1070.280 | 3736.294 | 3934.591 | 0.000 | 1197.375 | 1531.548 |

## Interpretation

The result is already useful because no policy dominates every metric:

1. `fifo` has a simple fairness story but misses many synthetic deadlines.
2. `shortest-cache` improves p50 and p95 latency but worsens p99 latency on this
   workload, which suggests starvation or delayed large requests.
3. `deadline` eliminates deadline misses and has the best throughput here, but
   it is optimizing deadlines that we generated synthetically.
4. Peak active KV cache does not monotonically improve under cache-aware
   scheduling because this simulator still uses slot concurrency rather than a
   true memory-capacity scheduler.

The follow-on capacity-aware baseline adds GPU capacity configs and
budget-relative memory metrics. That work is documented in
[`capacity-aware-scheduling.md`](capacity-aware-scheduling.md).
