# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r8`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `8`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 39.450 pp |
| Max cache-hit population stdev | 0.122 |
| Mean shared-minus-control direct counter hit rate | 95.797 pp |
| Direct counter hit-rate 90% bootstrap interval | 95.748 pp to 95.845 pp |
| Throughput-ratio delta 90% bootstrap interval | 9.015 to 10.711 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.957 to -0.924 |
| p95 latency-ratio delta 90% bootstrap interval | -0.925 to -0.897 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.736 to -0.632 |
| Max direct counter hit-rate population stdev | 0.093 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 32 | 8 | 0.400% | 0.446% | 304895.875 | 1360.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 32 | 8 | 39.850% | 96.243% | 115011.000 | 110690.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 8 | 39.850% | 0.400% | 95.797 pp | 95.748 pp to 95.845 pp | 9.863 | 9.015 to 10.711 | -0.942 | -0.957 to -0.924 | -0.911 | -0.925 to -0.897 | -0.683 | -0.736 to -0.632 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
