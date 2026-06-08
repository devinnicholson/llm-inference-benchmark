# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-smoke-r2`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `2`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 45.350 pp |
| Max cache-hit population stdev | 0.050 |
| Mean shared-minus-control direct counter hit rate | 92.605 pp |
| Direct counter hit-rate 90% bootstrap interval | 92.512 pp to 92.697 pp |
| Throughput-ratio delta 90% bootstrap interval | 7.333 to 7.352 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.915 to -0.915 |
| p95 latency-ratio delta 90% bootstrap interval | -0.908 to -0.888 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.948 to -0.947 |
| Max direct counter hit-rate population stdev | 0.092 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 16 | 2 | 3.300% | 0.446% | 57337.000 | 256.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 16 | 2 | 48.650% | 93.051% | 57405.000 | 53416.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 48.650% | 3.300% | 92.605 pp | 92.512 pp to 92.697 pp | 7.342 | 7.333 to 7.352 | -0.915 | -0.915 to -0.915 | -0.898 | -0.908 to -0.888 | -0.948 | -0.948 to -0.947 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
