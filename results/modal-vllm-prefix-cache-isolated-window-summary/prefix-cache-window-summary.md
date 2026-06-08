# Prefix-Cache Measured-Window Estimate

Source: `results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source`
Source mode: `vllm-prefix-cache-isolated-neutral-warmup`
Warmup runs: `1`
Warmup prompt profile: `neutral_long`

## Summary

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control estimated measured cache hit rate | 66.548 pp |
| Mean shared-minus-control logged after-window cache hit rate | 36.667 pp |

## Scenario Estimates

| Profile | Requests | Runs | Warmup hit | Logged after | Estimated measured |
| --- | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 1 | 1.700% | 2.200% | 2.596% |
| `matched_unique_prefix` | 4 | 1 | 2.500% | 2.600% | 2.679% |
| `matched_unique_prefix` | 8 | 1 | 3.000% | 2.800% | 2.641% |
| `shared_prefix_long` | 2 | 1 | 1.700% | 28.100% | 49.566% |
| `shared_prefix_long` | 4 | 1 | 2.500% | 41.400% | 73.113% |
| `shared_prefix_long` | 8 | 1 | 3.000% | 48.100% | 84.882% |

## Shared Vs Control

| Requests | Estimated Shared | Estimated Control | Delta | Throughput Delta | p95 Latency Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 49.566% | 2.596% | 46.970 pp | 2.822 | -0.567 |
| 4 | 73.113% | 2.679% | 70.434 pp | -0.986 | 0.478 |
| 8 | 84.882% | 2.641% | 82.241 pp | 0.074 | -0.039 |

Estimate method: token-weighted before/after rate delta. This is an approximation until direct vLLM hit/miss counters are captured.
