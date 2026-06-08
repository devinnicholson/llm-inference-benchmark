# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-no-repeat-scaling-smoke`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `2`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 31.625 pp |
| Max cache-hit population stdev | 0.050 |
| Mean shared-minus-control direct counter hit rate | 76.997 pp |
| Direct counter hit-rate 90% bootstrap interval | 75.781 pp to 78.163 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.022 to 0.590 |
| p95 latency-ratio delta 90% bootstrap interval | -0.293 to 0.033 |
| Max direct counter hit-rate population stdev | 0.122 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_no_repeat_variant` | 8 | 2 | 3.700% | 4.815% | 2658.500 | 128.000 |
| `matched_unique_prefix_no_repeat_variant` | 12 | 2 | 22.300% | 4.786% | 4011.500 | 192.000 |
| `matched_unique_prefix_no_repeat_variant` | 16 | 2 | 31.550% | 4.772% | 5364.500 | 256.000 |
| `matched_unique_prefix_no_repeat_variant` | 20 | 2 | 37.100% | 4.769% | 6710.500 | 320.000 |
| `shared_prefix_long_no_repeat_variant` | 8 | 2 | 33.650% | 78.532% | 2587.500 | 2032.000 |
| `shared_prefix_long_no_repeat_variant` | 12 | 2 | 53.800% | 81.547% | 3904.500 | 3184.000 |
| `shared_prefix_long_no_repeat_variant` | 16 | 2 | 63.850% | 83.041% | 5221.500 | 4336.000 |
| `shared_prefix_long_no_repeat_variant` | 20 | 2 | 69.850% | 84.011% | 6532.500 | 5488.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 33.650% | 3.700% | 73.717 pp | 73.617 pp to 73.817 pp | 0.061 | -0.085 to 0.207 | -0.079 | -0.241 to 0.084 |
| 12 | 2 | 53.800% | 22.300% | 76.761 pp | 76.653 pp to 76.869 pp | 0.039 | -0.197 to 0.276 | 0.000 | -0.218 to 0.219 |
| 16 | 2 | 63.850% | 31.550% | 78.269 pp | 78.157 pp to 78.382 pp | 0.680 | -0.126 to 1.487 | -0.273 | -0.656 to 0.110 |
| 20 | 2 | 69.850% | 37.100% | 79.242 pp | 79.127 pp to 79.358 pp | 0.164 | -0.080 to 0.408 | -0.131 | -0.345 to 0.084 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
