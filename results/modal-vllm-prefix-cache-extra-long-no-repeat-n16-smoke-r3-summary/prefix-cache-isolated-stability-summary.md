# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-smoke-r3`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_extra_long`
Repeats: `3`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 41.967 pp |
| Max cache-hit population stdev | 0.094 |
| Mean shared-minus-control direct counter hit rate | 89.891 pp |
| Direct counter hit-rate 90% bootstrap interval | 89.860 pp to 89.922 pp |
| Throughput-ratio delta 90% bootstrap interval | 0.671 to 0.998 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.671 to -0.565 |
| p95 latency-ratio delta 90% bootstrap interval | -1.107 to -0.401 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -1.114 to -0.485 |
| Max direct counter hit-rate population stdev | 0.033 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_extra_long_no_repeat_variant` | 16 | 3 | 9.667% | 1.342% | 19074.333 | 256.000 |
| `shared_prefix_extra_long_no_repeat_variant` | 16 | 3 | 51.633% | 91.233% | 19133.333 | 17456.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 51.633% | 9.667% | 89.891 pp | 89.860 pp to 89.922 pp | 0.834 | 0.671 to 0.998 | -0.618 | -0.671 to -0.565 | -0.754 | -1.107 to -0.401 | -0.800 | -1.114 to -0.485 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
