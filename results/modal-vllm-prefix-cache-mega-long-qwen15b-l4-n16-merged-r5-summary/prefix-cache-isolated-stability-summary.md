# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r5`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_mega_long`
Repeats: `5`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 45.380 pp |
| Max cache-hit population stdev | 0.075 |
| Mean shared-minus-control direct counter hit rate | 92.623 pp |
| Direct counter hit-rate 90% bootstrap interval | 92.549 pp to 92.698 pp |
| Throughput-ratio delta 90% bootstrap interval | 7.057 to 7.921 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.921 to -0.894 |
| p95 latency-ratio delta 90% bootstrap interval | -0.913 to -0.881 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.958 to -0.939 |
| Max direct counter hit-rate population stdev | 0.091 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_mega_long_no_repeat_variant` | 16 | 5 | 3.300% | 0.447% | 57325.800 | 256.000 |
| `shared_prefix_mega_long_no_repeat_variant` | 16 | 5 | 48.680% | 93.070% | 57470.600 | 53488.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 5 | 48.680% | 3.300% | 92.623 pp | 92.549 pp to 92.698 pp | 7.487 | 7.057 to 7.921 | -0.908 | -0.921 to -0.894 | -0.898 | -0.913 to -0.881 | -0.949 | -0.958 to -0.939 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
