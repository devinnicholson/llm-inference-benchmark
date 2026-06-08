# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-no-repeat-n16`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `3`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 32.700 pp |
| Max cache-hit population stdev | 0.309 |
| Mean shared-minus-control direct counter hit rate | 78.472 pp |
| Direct counter hit-rate 90% bootstrap interval | 78.232 pp to 78.713 pp |
| Throughput-ratio delta 90% bootstrap interval | 0.209 to 0.587 |
| p95 latency-ratio delta 90% bootstrap interval | -0.432 to -0.191 |
| Max direct counter hit-rate population stdev | 0.206 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_no_repeat_variant` | 16 | 3 | 31.333% | 4.698% | 5452.333 | 256.000 |
| `shared_prefix_long_no_repeat_variant` | 16 | 3 | 64.033% | 83.170% | 5309.333 | 4416.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 64.033% | 31.333% | 78.472 pp | 78.232 pp to 78.713 pp | 0.398 | 0.209 to 0.587 | -0.311 | -0.432 to -0.191 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
