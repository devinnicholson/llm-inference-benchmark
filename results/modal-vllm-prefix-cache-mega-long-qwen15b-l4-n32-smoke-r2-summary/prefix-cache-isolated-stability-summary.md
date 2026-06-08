# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-smoke-r2`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `2`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 39.400 pp |
| Max cache-hit population stdev | 0.100 |
| Mean shared-minus-control direct counter hit rate | 95.772 pp |
| Direct counter hit-rate 90% bootstrap interval | 95.677 pp to 95.868 pp |
| Throughput-ratio delta 90% bootstrap interval | 9.619 to 9.815 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.974 to -0.940 |
| p95 latency-ratio delta 90% bootstrap interval | -0.946 to -0.910 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.677 to -0.672 |
| Max direct counter hit-rate population stdev | 0.095 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 32 | 2 | 0.400% | 0.446% | 304972.500 | 1360.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 32 | 2 | 39.800% | 96.218% | 114847.000 | 110504.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2 | 39.800% | 0.400% | 95.772 pp | 95.677 pp to 95.868 pp | 9.717 | 9.619 to 9.815 | -0.957 | -0.974 to -0.940 | -0.928 | -0.946 to -0.910 | -0.675 | -0.677 to -0.672 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
