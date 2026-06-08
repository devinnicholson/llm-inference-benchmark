# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-variant-n16`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `3`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 15.700 pp |
| Max cache-hit population stdev | 0.262 |
| Mean shared-minus-control direct counter hit rate | 38.250 pp |
| Direct counter hit-rate 90% bootstrap interval | 38.162 pp to 38.338 pp |
| Throughput-ratio delta 90% bootstrap interval | 0.108 to 0.422 |
| p95 latency-ratio delta 90% bootstrap interval | -0.212 to -0.082 |
| Max direct counter hit-rate population stdev | 0.114 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_variant` | 16 | 3 | 50.533% | 50.525% | 5404.667 | 2730.667 |
| `shared_prefix_long_variant` | 16 | 3 | 66.233% | 88.775% | 5262.667 | 4672.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 66.233% | 50.533% | 38.250 pp | 38.162 pp to 38.338 pp | 0.265 | 0.108 to 0.422 | -0.147 | -0.212 to -0.082 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
