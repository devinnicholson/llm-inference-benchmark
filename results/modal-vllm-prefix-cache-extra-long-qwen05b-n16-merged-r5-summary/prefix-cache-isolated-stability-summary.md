# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-merged-r5`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_extra_long`
Repeats: `5`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 41.880 pp |
| Max cache-hit population stdev | 0.232 |
| Mean shared-minus-control direct counter hit rate | 90.145 pp |
| Direct counter hit-rate 90% bootstrap interval | 89.869 pp to 90.421 pp |
| Throughput-ratio delta 90% bootstrap interval | 1.276 to 2.210 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.848 to -0.777 |
| p95 latency-ratio delta 90% bootstrap interval | -0.909 to -0.392 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.763 to -0.603 |
| Max direct counter hit-rate population stdev | 0.353 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_extra_long_no_repeat_variant` | 16 | 5 | 9.800% | 1.417% | 18072.200 | 256.000 |
| `shared_prefix_extra_long_no_repeat_variant` | 16 | 5 | 51.680% | 91.562% | 17998.600 | 16480.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 5 | 51.680% | 9.800% | 90.145 pp | 89.869 pp to 90.421 pp | 1.747 | 1.276 to 2.210 | -0.811 | -0.848 to -0.777 | -0.633 | -0.909 to -0.392 | -0.692 | -0.763 to -0.603 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
