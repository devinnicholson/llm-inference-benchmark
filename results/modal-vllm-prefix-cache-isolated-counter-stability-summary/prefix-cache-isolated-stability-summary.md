# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `3`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 36.667 pp |
| Max cache-hit population stdev | 0.000 |
| Mean shared-minus-control direct counter hit rate | 66.501 pp |
| Direct counter hit-rate 90% bootstrap interval | 58.660 pp to 74.342 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.160 to 0.576 |
| p95 latency-ratio delta 90% bootstrap interval | -0.181 to 0.095 |
| Max direct counter hit-rate population stdev | 0.000 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 3 | 2.200% | 2.682% | 1193.000 | 32.000 |
| `matched_unique_prefix` | 4 | 3 | 2.600% | 2.682% | 2386.000 | 64.000 |
| `matched_unique_prefix` | 8 | 3 | 2.800% | 2.686% | 4765.000 | 128.000 |
| `shared_prefix_long` | 2 | 3 | 28.100% | 49.612% | 1161.000 | 576.000 |
| `shared_prefix_long` | 4 | 3 | 41.400% | 73.040% | 2322.000 | 1696.000 |
| `shared_prefix_long` | 8 | 3 | 48.100% | 84.901% | 4636.000 | 3936.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 28.100% | 2.200% | 46.930 pp | 46.930 pp to 46.930 pp | 0.660 | -0.019 to 1.339 | -0.215 | -0.451 to 0.021 |
| 4 | 3 | 41.400% | 2.600% | 70.358 pp | 70.358 pp to 70.358 pp | -0.280 | -0.500 to -0.060 | 0.179 | 0.052 to 0.305 |
| 8 | 3 | 48.100% | 2.800% | 82.215 pp | 82.215 pp to 82.215 pp | 0.085 | 0.048 to 0.121 | -0.078 | -0.114 to -0.042 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
