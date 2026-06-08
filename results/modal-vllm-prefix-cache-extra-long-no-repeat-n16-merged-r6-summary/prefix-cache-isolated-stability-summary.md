# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_extra_long`
Repeats: `6`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 41.967 pp |
| Max cache-hit population stdev | 0.094 |
| Mean shared-minus-control direct counter hit rate | 89.891 pp |
| Direct counter hit-rate 90% bootstrap interval | 89.867 pp to 89.915 pp |
| Throughput-ratio delta 90% bootstrap interval | 0.411 to 1.111 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.701 to -0.376 |
| p95 latency-ratio delta 90% bootstrap interval | -0.867 to -0.235 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.888 to -0.356 |
| Max direct counter hit-rate population stdev | 0.033 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_extra_long_no_repeat_variant` | 16 | 6 | 9.667% | 1.342% | 19074.333 | 256.000 |
| `shared_prefix_extra_long_no_repeat_variant` | 16 | 6 | 51.633% | 91.233% | 19133.333 | 17456.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 6 | 51.633% | 9.667% | 89.891 pp | 89.867 pp to 89.915 pp | 0.772 | 0.411 to 1.111 | -0.561 | -0.701 to -0.376 | -0.525 | -0.867 to -0.235 | -0.583 | -0.888 to -0.356 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
