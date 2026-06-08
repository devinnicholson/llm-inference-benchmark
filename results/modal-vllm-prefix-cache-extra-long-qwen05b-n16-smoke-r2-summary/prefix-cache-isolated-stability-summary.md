# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-smoke-r2`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_extra_long`
Repeats: `2`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 41.850 pp |
| Max cache-hit population stdev | 0.250 |
| Mean shared-minus-control direct counter hit rate | 90.099 pp |
| Direct counter hit-rate 90% bootstrap interval | 89.716 pp to 90.482 pp |
| Throughput-ratio delta 90% bootstrap interval | 1.913 to 2.768 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.896 to -0.779 |
| p95 latency-ratio delta 90% bootstrap interval | -1.285 to -0.582 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.801 to -0.684 |
| Max direct counter hit-rate population stdev | 0.382 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_extra_long_no_repeat_variant` | 16 | 2 | 9.800% | 1.419% | 18045.000 | 256.000 |
| `shared_prefix_extra_long_no_repeat_variant` | 16 | 2 | 51.650% | 91.518% | 17981.000 | 16456.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2 | 51.650% | 9.800% | 90.099 pp | 89.716 pp to 90.482 pp | 2.341 | 1.913 to 2.768 | -0.838 | -0.896 to -0.779 | -0.934 | -1.285 to -0.582 | -0.742 | -0.801 to -0.684 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
