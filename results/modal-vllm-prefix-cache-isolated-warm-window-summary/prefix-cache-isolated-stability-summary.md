# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-isolated-warm-window`
Source mode: `vllm-prefix-cache-isolated-warm-window`
Warmup runs: `1`
Repeats: `1`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 33.233 pp |
| Max cache-hit population stdev | 0.000 |

## Scenario Stability

| Profile | Requests | Runs | Cache Hit Mean | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 1 | 50.300% | 50.300% | 50.300% | 0.000 |
| `matched_unique_prefix` | 4 | 1 | 50.600% | 50.600% | 50.600% | 0.000 |
| `matched_unique_prefix` | 8 | 1 | 50.900% | 50.900% | 50.900% | 0.000 |
| `shared_prefix_long` | 2 | 1 | 73.700% | 73.700% | 73.700% | 0.000 |
| `shared_prefix_long` | 4 | 1 | 85.800% | 85.800% | 85.800% | 0.000 |
| `shared_prefix_long` | 8 | 1 | 92.000% | 92.000% | 92.000% | 0.000 |

## Shared Vs Control

| Requests | Shared Hit Mean | Control Hit Mean | Delta | Throughput Delta | p95 Latency Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 73.700% | 50.300% | 23.400 pp | 2.605 | -0.700 |
| 4 | 85.800% | 50.600% | 35.200 pp | -0.359 | 0.172 |
| 8 | 92.000% | 50.900% | 41.100 pp | 0.433 | -0.517 |
