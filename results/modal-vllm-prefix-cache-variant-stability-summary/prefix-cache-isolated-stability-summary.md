# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-variant-stability`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`
Repeats: `3`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 24.522 pp |
| Max cache-hit population stdev | 0.497 |
| Mean shared-minus-control direct counter hit rate | 59.790 pp |
| Direct counter hit-rate 90% bootstrap interval | 52.709 pp to 66.830 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.112 to 0.569 |
| p95 latency-ratio delta 90% bootstrap interval | -0.148 to 0.149 |
| Max direct counter hit-rate population stdev | 0.174 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_variant` | 2 | 3 | 2.967% | 4.727% | 677.333 | 32.000 |
| `matched_unique_prefix_variant` | 4 | 3 | 3.467% | 4.731% | 1353.333 | 64.000 |
| `matched_unique_prefix_variant` | 8 | 3 | 3.700% | 4.739% | 2702.333 | 128.000 |
| `shared_prefix_long_variant` | 2 | 3 | 20.300% | 46.915% | 659.333 | 309.333 |
| `shared_prefix_long_variant` | 4 | 3 | 29.400% | 68.013% | 1317.333 | 896.000 |
| `shared_prefix_long_variant` | 8 | 3 | 34.000% | 78.639% | 2631.333 | 2069.333 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 20.300% | 2.967% | 42.188 pp | 42.065 pp to 42.311 pp | 0.474 | -0.244 to 1.192 | -0.016 | -0.380 to 0.348 |
| 4 | 3 | 29.400% | 3.467% | 63.282 pp | 63.097 pp to 63.467 pp | -0.045 | -0.070 to -0.020 | 0.052 | 0.022 to 0.082 |
| 8 | 3 | 34.000% | 3.700% | 73.899 pp | 73.683 pp to 74.115 pp | 0.000 | -0.015 to 0.016 | -0.002 | -0.018 to 0.015 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
