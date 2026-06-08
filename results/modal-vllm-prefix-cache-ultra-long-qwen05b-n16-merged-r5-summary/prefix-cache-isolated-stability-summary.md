# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-ultra-long-qwen05b-n16-merged-r5`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_ultra_long`
Repeats: `5`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 43.900 pp |
| Max cache-hit population stdev | 0.110 |
| Mean shared-minus-control direct counter hit rate | 91.472 pp |
| Direct counter hit-rate 90% bootstrap interval | 91.372 pp to 91.581 pp |
| Throughput-ratio delta 90% bootstrap interval | 1.387 to 2.967 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -1.372 to -0.848 |
| p95 latency-ratio delta 90% bootstrap interval | -1.063 to -0.601 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.857 to -0.629 |
| Max direct counter hit-rate population stdev | 0.131 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_ultra_long_no_repeat_variant` | 16 | 5 | 5.900% | 0.822% | 31145.000 | 256.000 |
| `shared_prefix_ultra_long_no_repeat_variant` | 16 | 5 | 49.800% | 92.294% | 31169.800 | 28768.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 5 | 49.800% | 5.900% | 91.472 pp | 91.372 pp to 91.581 pp | 2.184 | 1.387 to 2.967 | -1.032 | -1.372 to -0.848 | -0.809 | -1.063 to -0.601 | -0.773 | -0.857 to -0.629 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
