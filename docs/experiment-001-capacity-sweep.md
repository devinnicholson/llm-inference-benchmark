# Experiment 001: Capacity Sweep

## Question

When KV-cache memory is the bottleneck, do simple scheduler policies trade off
deadline behavior, throughput, and budget violations in measurable ways?

This experiment turns the one-off capacity baseline into a reproducible local
sweep. The point is not to claim production-serving behavior yet. The point is
to make the experiment matrix, result files, and first observation concrete
enough that the next backend can be swapped in without changing the research
question.

## Method

Run the deterministic generated workload across:

- Workload: `workloads/generated/mixed_bursty_32_seed568.json`
- Model config: `configs/models/llama-7b-gqa-fp16.json`
- Capacity configs:
  - `configs/capacity/tight-1gb-kv.json`
  - `configs/capacity/tight-1-2gb-kv.json`
  - `configs/capacity/a10g-24gb-7b-gqa.json`
- Concurrent request slots: `1`, `2`, `4`, `8`
- Policies: `fifo`, `shortest-cache`, `deadline`, `memory-aware-deadline`

Command:

```bash
python3 scripts/run_sweep.py
```

Outputs:

- JSON: `results/experiment-001-capacity-sweep/sweep-results.json`
- CSV: `results/experiment-001-capacity-sweep/sweep-results.csv`

## Four-Slot Result

The four-slot setting is the first interesting point. One or two slots underuse
the synthetic server, while eight slots creates enough overlap that prompt-only
admission control becomes visibly incomplete.

| Capacity | Policy | p95 latency ms | Peak KV MiB | p95 KV MiB | Budget exceeded ms | Deadline miss rate | Output tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `tight-1gb-kv` | `fifo` | 3576.177 | 1148.250 | 1046.250 | 322.343 | 0.438 | 1459.600 |
| `tight-1gb-kv` | `shortest-cache` | 3471.038 | 1397.250 | 1300.000 | 465.071 | 0.219 | 1388.002 |
| `tight-1gb-kv` | `deadline` | 3736.294 | 1197.375 | 1150.875 | 625.775 | 0.000 | 1531.548 |
| `tight-1gb-kv` | `memory-aware-deadline` | 3736.294 | 1067.750 | 1021.250 | 197.510 | 0.000 | 1531.548 |
| `tight-1-2gb-kv` | `fifo` | 3576.177 | 1148.250 | 1046.250 | 0.000 | 0.438 | 1459.600 |
| `tight-1-2gb-kv` | `shortest-cache` | 3471.038 | 1397.250 | 1300.000 | 414.308 | 0.219 | 1388.002 |
| `tight-1-2gb-kv` | `deadline` | 3736.294 | 1197.375 | 1150.875 | 0.000 | 0.000 | 1531.548 |
| `tight-1-2gb-kv` | `memory-aware-deadline` | 3736.294 | 1197.375 | 1150.875 | 0.000 | 0.000 | 1531.548 |
| `a10g-24gb-7b-gqa` | `fifo` | 3576.177 | 1148.250 | 1046.250 | 0.000 | 0.438 | 1459.600 |
| `a10g-24gb-7b-gqa` | `shortest-cache` | 3471.038 | 1397.250 | 1300.000 | 0.000 | 0.219 | 1388.002 |
| `a10g-24gb-7b-gqa` | `deadline` | 3736.294 | 1197.375 | 1150.875 | 0.000 | 0.000 | 1531.548 |
| `a10g-24gb-7b-gqa` | `memory-aware-deadline` | 3736.294 | 1197.375 | 1150.875 | 0.000 | 0.000 | 1531.548 |

## Interpretation

The tight 1 GiB budget is the useful stress case. At four slots,
`memory-aware-deadline` keeps the synthetic deadline miss rate at zero while
reducing budget-exceeded time from 625.775 ms to 197.510 ms versus plain
`deadline`. That is a concrete result worth preserving.

The result is also narrow. The memory-aware policy only checks whether the
candidate request's prompt KV allocation fits at admission time. It does not
reserve the request's future decode growth against the active timeline. That is
why budget violations remain.

The 1.2 GiB and A10G-style budgets show the control side of the experiment. When
capacity is less constrained, `memory-aware-deadline` converges to `deadline`
because no prompt admission conflict exists. That is expected and helps validate
that the policy is only changing behavior under memory pressure.

## Eight-Slot Stress

At eight slots under `tight-1gb-kv`, `memory-aware-deadline` reduces
budget-exceeded time relative to `deadline`, but throughput drops:

| Policy | p95 latency ms | Peak KV MiB | Budget exceeded ms | Deadline miss rate | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `deadline` | 2271.541 | 1981.500 | 1056.915 | 0.000 | 2420.810 |
| `memory-aware-deadline` | 2583.532 | 1551.250 | 632.773 | 0.000 | 2118.366 |

This is the clearest next research opening: the policy is memory-aware, but it
is not yet memory-reserving. A stronger scheduler should reason about future KV
growth during decode, not only prompt allocation.

## Next Step

The next implementation should add a reservation-aware policy. A reasonable
first version would estimate the full request KV lifetime, reject candidates
whose projected timeline exceeds capacity, and compare that policy against
`memory-aware-deadline` on the same sweep.
