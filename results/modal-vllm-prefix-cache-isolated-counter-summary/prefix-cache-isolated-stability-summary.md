# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters`
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
| Mean shared-minus-control direct counter hit rate | 66.501 pp |
| Max direct counter hit-rate population stdev | 0.000 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 1 | 2.200% | 2.682% | 1193.000 | 32.000 |
| `matched_unique_prefix` | 4 | 1 | 2.600% | 2.682% | 2386.000 | 64.000 |
| `matched_unique_prefix` | 8 | 1 | 2.800% | 2.686% | 4765.000 | 128.000 |
| `shared_prefix_long` | 2 | 1 | 28.100% | 49.612% | 1161.000 | 576.000 |
| `shared_prefix_long` | 4 | 1 | 41.400% | 73.040% | 2322.000 | 1696.000 |
| `shared_prefix_long` | 8 | 1 | 48.100% | 84.901% | 4636.000 | 3936.000 |

## Shared Vs Control

| Requests | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Throughput Delta | p95 Latency Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 28.100% | 2.200% | 46.930 pp | 2.480 | -0.729 |
| 4 | 41.400% | 2.600% | 70.358 pp | -0.539 | 0.357 |
| 8 | 48.100% | 2.800% | 82.215 pp | 0.048 | -0.065 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
