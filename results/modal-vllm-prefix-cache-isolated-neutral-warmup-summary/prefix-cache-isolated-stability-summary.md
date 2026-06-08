# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-isolated-neutral-warmup`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `1`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 36.667 pp |
| Max cache-hit population stdev | 0.000 |

## Scenario Stability

| Profile | Requests | Runs | Cache Hit Mean | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 1 | 2.200% | 2.200% | 2.200% | 0.000 |
| `matched_unique_prefix` | 4 | 1 | 2.600% | 2.600% | 2.600% | 0.000 |
| `matched_unique_prefix` | 8 | 1 | 2.800% | 2.800% | 2.800% | 0.000 |
| `shared_prefix_long` | 2 | 1 | 28.100% | 28.100% | 28.100% | 0.000 |
| `shared_prefix_long` | 4 | 1 | 41.400% | 41.400% | 41.400% | 0.000 |
| `shared_prefix_long` | 8 | 1 | 48.100% | 48.100% | 48.100% | 0.000 |

## Shared Vs Control

| Requests | Shared Hit Mean | Control Hit Mean | Delta | Throughput Delta | p95 Latency Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 28.100% | 2.200% | 25.900 pp | 0.148 | -0.016 |
| 4 | 41.400% | 2.600% | 38.800 pp | -0.740 | 0.435 |
| 8 | 48.100% | 2.800% | 45.300 pp | 0.035 | -0.034 |
