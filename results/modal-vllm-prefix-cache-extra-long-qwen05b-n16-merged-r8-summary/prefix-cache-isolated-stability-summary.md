# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-merged-r8`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_extra_long`
Repeats: `8`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 41.888 pp |
| Max cache-hit population stdev | 0.226 |
| Mean shared-minus-control direct counter hit rate | 90.157 pp |
| Direct counter hit-rate 90% bootstrap interval | 89.965 pp to 90.348 pp |
| Throughput-ratio delta 90% bootstrap interval | 1.199 to 1.892 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.805 to -0.705 |
| p95 latency-ratio delta 90% bootstrap interval | -0.776 to -0.440 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.739 to -0.638 |
| Max direct counter hit-rate population stdev | 0.345 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_extra_long_no_repeat_variant` | 16 | 8 | 9.800% | 1.416% | 18079.000 | 256.000 |
| `shared_prefix_extra_long_no_repeat_variant` | 16 | 8 | 51.688% | 91.573% | 18003.000 | 16486.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 51.688% | 9.800% | 90.157 pp | 89.965 pp to 90.348 pp | 1.541 | 1.199 to 1.892 | -0.757 | -0.805 to -0.705 | -0.597 | -0.776 to -0.440 | -0.695 | -0.739 to -0.638 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
