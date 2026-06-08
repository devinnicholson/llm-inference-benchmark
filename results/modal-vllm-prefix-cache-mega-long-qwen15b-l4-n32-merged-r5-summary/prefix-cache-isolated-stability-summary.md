# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r5`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `5`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 39.440 pp |
| Max cache-hit population stdev | 0.120 |
| Mean shared-minus-control direct counter hit rate | 95.792 pp |
| Direct counter hit-rate 90% bootstrap interval | 95.715 pp to 95.869 pp |
| Throughput-ratio delta 90% bootstrap interval | 8.764 to 10.618 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.962 to -0.915 |
| p95 latency-ratio delta 90% bootstrap interval | -0.937 to -0.904 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.786 to -0.686 |
| Max direct counter hit-rate population stdev | 0.094 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 32 | 5 | 0.400% | 0.446% | 304909.400 | 1360.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 32 | 5 | 39.840% | 96.238% | 114978.200 | 110652.800 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 5 | 39.840% | 0.400% | 95.792 pp | 95.715 pp to 95.869 pp | 9.711 | 8.764 to 10.618 | -0.941 | -0.962 to -0.915 | -0.922 | -0.937 to -0.904 | -0.735 | -0.786 to -0.686 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
