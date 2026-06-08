# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r8`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `8`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 45.388 pp |
| Max cache-hit population stdev | 0.078 |
| Mean shared-minus-control direct counter hit rate | 92.628 pp |
| Direct counter hit-rate 90% bootstrap interval | 92.581 pp to 92.675 pp |
| Throughput-ratio delta 90% bootstrap interval | 7.222 to 7.795 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.918 to -0.900 |
| p95 latency-ratio delta 90% bootstrap interval | -0.903 to -0.881 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.954 to -0.941 |
| Max direct counter hit-rate population stdev | 0.090 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 16 | 8 | 3.300% | 0.447% | 57323.000 | 256.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 16 | 8 | 48.688% | 93.075% | 57487.000 | 53506.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 48.688% | 3.300% | 92.628 pp | 92.581 pp to 92.675 pp | 7.491 | 7.222 to 7.795 | -0.910 | -0.918 to -0.900 | -0.892 | -0.903 to -0.881 | -0.947 | -0.954 to -0.941 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
