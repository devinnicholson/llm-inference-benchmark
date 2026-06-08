# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `8`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 32.600 pp |
| Max cache-hit population stdev | 0.285 |
| Mean shared-minus-control direct counter hit rate | 78.422 pp |
| Direct counter hit-rate 90% bootstrap interval | 78.269 pp to 78.602 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.332 to 0.185 |
| p95 latency-ratio delta 90% bootstrap interval | -0.180 to 0.023 |
| Max direct counter hit-rate population stdev | 0.197 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_no_repeat_variant` | 16 | 8 | 31.387% | 4.716% | 5430.375 | 256.000 |
| `shared_prefix_long_no_repeat_variant` | 16 | 8 | 63.987% | 83.138% | 5287.375 | 4396.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 63.987% | 31.387% | 78.422 pp | 78.269 pp to 78.602 pp | -0.036 | -0.332 to 0.185 | -0.074 | -0.180 to 0.023 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
