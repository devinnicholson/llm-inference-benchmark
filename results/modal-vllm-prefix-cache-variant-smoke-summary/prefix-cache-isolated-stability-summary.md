# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-variant-smoke`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `2`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 25.600 pp |
| Max cache-hit population stdev | 0.000 |
| Mean shared-minus-control direct counter hit rate | 63.121 pp |
| Direct counter hit-rate 90% bootstrap interval | 63.048 pp to 63.194 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.167 to -0.091 |
| p95 latency-ratio delta 90% bootstrap interval | 0.097 to 0.252 |
| Max direct counter hit-rate population stdev | 0.079 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_variant` | 4 | 2 | 3.500% | 4.807% | 1331.500 | 64.000 |
| `shared_prefix_long_variant` | 4 | 2 | 29.100% | 67.928% | 1295.500 | 880.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 29.100% | 3.500% | 63.121 pp | 63.048 pp to 63.194 pp | -0.129 | -0.167 to -0.091 | 0.174 | 0.097 to 0.252 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
