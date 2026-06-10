# KV-Budget Request-Count Comparison

GPU memory utilization: `0.325`
Low-n source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n16-batched-tokens60640-seed4107-pressure-curve-r1/server-cache-pressure-curve.json`
High-n source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-gpu0325-n32-batched-tokens60640-seed3805-seed3906-pressure-curve-r1/server-cache-pressure-curve.json`

## Headline

At the same KV-budget floor, shared-prefix throughput rises from 4.619x at n=16 to 10.741x at n=32.

The matched-unique control remains near neutral over the same request-count change: 0.938x to 0.975x.

## Rows

| Source | Profile | n | Trials | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n16_seed4107` | `matched_unique` | 16 | 1 | 4.036 | 0.438% | 0.938x | 1.066x |
| `n32_seed3805_seed3906` | `matched_unique` | 32 | 2 | 8.080 | 0.442% | 0.975x | 1.026x |
| `n16_seed4107` | `shared_prefix` | 16 | 1 | 4.053 | 93.116% | 4.619x | 0.216x |
| `n32_seed3805_seed3906` | `shared_prefix` | 32 | 2 | 8.091 | 96.204% | 10.741x | 0.095x |

## Request-Count Delta

| Profile | n Change | Pressure Ratio | Throughput Delta | Throughput Ratio Change | p95 Latency Delta | Hit Rate Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique` | 16 -> 32 | 2.002x | 0.037x | 1.040x | -0.040x | 0.005 pp |
| `shared_prefix` | 16 -> 32 | 1.996x | 6.121x | 2.325x | -0.121x | 3.089 pp |

## Notes

This comparison isolates request count at the lowest successful KV-budget point. The n=16 row is a one-seed probe; the n=32 row is the existing two-seed replicated floor point.
