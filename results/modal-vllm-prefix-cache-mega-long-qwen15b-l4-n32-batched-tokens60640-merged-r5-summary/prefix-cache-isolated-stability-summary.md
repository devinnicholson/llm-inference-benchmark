# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-batched-tokens60640-merged-r5`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `5`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 47.000 pp |
| Max cache-hit population stdev | 0.110 |
| Mean shared-minus-control direct counter hit rate | 95.830 pp |
| Direct counter hit-rate 90% bootstrap interval | 95.754 pp to 95.869 pp |
| Throughput-ratio delta 90% bootstrap interval | 6.779 to 9.648 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.957 to -0.940 |
| p95 latency-ratio delta 90% bootstrap interval | -0.923 to -0.892 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.955 to -0.904 |
| Max direct counter hit-rate population stdev | 0.077 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 32 | 5 | 4.700% | 0.446% | 114767.000 | 512.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 32 | 5 | 51.700% | 96.277% | 115138.200 | 110851.200 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5 | 51.700% | 4.700% | 95.830 pp | 95.754 pp to 95.869 pp | 8.343 | 6.779 to 9.648 | -0.949 | -0.957 to -0.940 | -0.909 | -0.923 to -0.892 | -0.931 | -0.955 to -0.904 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
