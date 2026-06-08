# Prefix-Cache Isolated Stability Summary

Source: `results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8`
Source mode: `vllm-prefix-cache-isolated-merge`
Warmup runs: `1`
Warmup prompt profile: `neutral_extra_long`
Repeats: `8`
Phase order: `cold_first`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 41.950 pp |
| Max cache-hit population stdev | 0.097 |
| Mean shared-minus-control direct counter hit rate | 89.885 pp |
| Direct counter hit-rate 90% bootstrap interval | 89.867 pp to 89.909 pp |
| Throughput-ratio delta 90% bootstrap interval | 0.489 to 1.011 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.694 to -0.445 |
| p95 latency-ratio delta 90% bootstrap interval | -0.747 to -0.284 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.791 to -0.399 |
| Max direct counter hit-rate population stdev | 0.032 |

## Scenario Stability

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_extra_long_no_repeat_variant` | 16 | 8 | 9.675% | 1.343% | 19056.375 | 256.000 |
| `shared_prefix_extra_long_no_repeat_variant` | 16 | 8 | 51.625% | 91.229% | 19123.375 | 17446.000 |

## Shared Vs Control

| Requests | Paired Obs | Logged Shared Hit Mean | Logged Control Hit Mean | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 First-Event Delta | p95 First-Event 90% CI | p95 Latency Delta | p95 Latency 90% CI | p95 Stream TPOT Delta | p95 Stream TPOT 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 51.625% | 9.675% | 89.885 pp | 89.867 pp to 89.909 pp | 0.761 | 0.489 to 1.011 | -0.589 | -0.694 to -0.445 | -0.500 | -0.747 to -0.284 | -0.567 | -0.791 to -0.399 |

Delta uses direct measured-window counters when available; logged hit means remain cumulative over warmup plus measured scenario.
