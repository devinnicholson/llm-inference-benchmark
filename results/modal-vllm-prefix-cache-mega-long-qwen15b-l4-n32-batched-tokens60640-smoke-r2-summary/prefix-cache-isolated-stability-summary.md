# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-batched-tokens60640-smoke-r2`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `2`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 47.050 pp |
| Max cache-hit population stdev | 0.050 |
| Mean shared-minus-control direct counter hit rate | 95.869 pp |
| Direct counter hit-rate 90% bootstrap interval | 95.868 pp to 95.870 pp |
| Throughput-ratio delta 90% bootstrap interval | 9.572 to 10.485 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.966 to -0.956 |
| p95 latency-ratio delta 90% bootstrap interval | -0.930 to -0.914 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.963 to -0.947 |
| Max direct counter hit-rate population stdev | 0.001 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 32 | 2 | 4.700% | 0.446% | 114799.000 | 512.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 32 | 2 | 51.750% | 96.315% | 115247.000 | 111000.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 51.750% | 4.700% | 95.869 pp | 95.868 pp to 95.870 pp | 10.029 | 9.572 to 10.485 | -0.961 | -0.966 to -0.956 | -0.922 | -0.930 to -0.914 | -0.955 | -0.963 to -0.947 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
