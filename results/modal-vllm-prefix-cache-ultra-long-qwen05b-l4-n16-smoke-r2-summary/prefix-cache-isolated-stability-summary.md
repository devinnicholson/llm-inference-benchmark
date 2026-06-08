# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-ultra-long-qwen05b-l4-n16-smoke-r2`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_ultra_long`
Repeats: `2`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 43.950 pp |
| Max cache-hit population stdev | 0.050 |
| Mean shared-minus-control direct counter hit rate | 91.523 pp |
| Direct counter hit-rate 90% bootstrap interval | 91.426 pp to 91.619 pp |
| Throughput-ratio delta 90% bootstrap interval | 1.358 to 5.178 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.965 to -0.886 |
| p95 latency-ratio delta 90% bootstrap interval | -0.855 to -0.620 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.841 to -0.749 |
| Max direct counter hit-rate population stdev | 0.095 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_ultra_long_no_repeat_variant` | 16 | 2 | 5.900% | 0.822% | 31161.000 | 256.000 |
| `shared_prefix_ultra_long_no_repeat_variant` | 16 | 2 | 49.850% | 92.344% | 31205.000 | 28816.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 49.850% | 5.900% | 91.523 pp | 91.426 pp to 91.619 pp | 3.268 | 1.358 to 5.178 | -0.925 | -0.965 to -0.886 | -0.738 | -0.855 to -0.620 | -0.795 | -0.841 to -0.749 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
