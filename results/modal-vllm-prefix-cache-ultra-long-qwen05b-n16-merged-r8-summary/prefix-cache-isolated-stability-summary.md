# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-ultra-long-qwen05b-n16-merged-r8`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_ultra_long`
Repeats: `8`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 43.888 pp |
| Max cache-hit population stdev | 0.117 |
| Mean shared-minus-control direct counter hit rate | 91.460 pp |
| Direct counter hit-rate 90% bootstrap interval | 91.377 pp to 91.547 pp |
| Throughput-ratio delta 90% bootstrap interval | 1.851 to 5.324 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -1.163 to -0.836 |
| p95 latency-ratio delta 90% bootstrap interval | -0.947 to -0.652 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.857 to -0.709 |
| Max direct counter hit-rate population stdev | 0.136 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_ultra_long_no_repeat_variant` | 16 | 8 | 5.900% | 0.822% | 31141.000 | 256.000 |
| `shared_prefix_ultra_long_no_repeat_variant` | 16 | 8 | 49.788% | 92.282% | 31161.000 | 28756.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 49.788% | 5.900% | 91.460 pp | 91.377 pp to 91.547 pp | 3.308 | 1.851 to 5.324 | -0.964 | -1.163 to -0.836 | -0.795 | -0.947 to -0.652 | -0.791 | -0.857 to -0.709 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
