# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-isolated-metrics-repeated`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 66.500 pp |
| Max cache-hit population stdev | 0.000 |

## Scenario Stability

| Profile | Requests | Runs | Cache Hit Mean | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 2 | 1.300% | 1.300% | 1.300% | 0.000 |
| `matched_unique_prefix` | 4 | 2 | 2.000% | 2.000% | 2.000% | 0.000 |
| `matched_unique_prefix` | 8 | 2 | 2.400% | 2.400% | 2.400% | 0.000 |
| `shared_prefix_long` | 2 | 2 | 48.200% | 48.200% | 48.200% | 0.000 |
| `shared_prefix_long` | 4 | 2 | 72.400% | 72.400% | 72.400% | 0.000 |
| `shared_prefix_long` | 8 | 2 | 84.600% | 84.600% | 84.600% | 0.000 |

## Shared Vs Control

| Requests | Shared Hit Mean | Control Hit Mean | Delta | Throughput Delta | p95 Latency Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 48.200% | 1.300% | 46.900 pp | 1.744 | -0.294 |
| 4 | 72.400% | 2.000% | 70.400 pp | -0.348 | 0.264 |
| 8 | 84.600% | 2.400% | 82.200 pp | 0.103 | -0.114 |
