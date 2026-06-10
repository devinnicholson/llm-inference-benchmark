# KV-Budget Request-Count Comparison

GPU memory utilization: `0.325`
Low-n source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n16-batched-tokens60640-seed4107-seed4208-pressure-curve-r1/server-cache-pressure-curve.json`
High-n source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-gpu0325-n32-batched-tokens60640-seed3805-seed3906-pressure-curve-r1/server-cache-pressure-curve.json`

## Headline

At the same KV-budget floor, shared-prefix throughput rises from 5.991x at n=16 to 10.741x at n=32.

The matched-unique control remains near neutral over the same request-count change: 0.994x to 0.975x.

## Rows

| Source | Profile | n | Trials | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `n16_seed4107_seed4208` | `matched_unique` | 16 | 2 | 4.036 | 0.438% | 0.994x | 1.009x |
| `n32_seed3805_seed3906` | `matched_unique` | 32 | 2 | 8.080 | 0.442% | 0.975x | 1.026x |
| `n16_seed4107_seed4208` | `shared_prefix` | 16 | 2 | 4.062 | 93.117% | 5.991x | 0.176x |
| `n32_seed3805_seed3906` | `shared_prefix` | 32 | 2 | 8.091 | 96.204% | 10.741x | 0.095x |

## Request-Count Delta

| Profile | n Change | Pressure Ratio | Throughput Delta | Throughput Ratio Change | p95 Latency Delta | Hit Rate Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique` | 16 -> 32 | 2.002x | -0.019x | 0.981x | 0.016x | 0.004 pp |
| `shared_prefix` | 16 -> 32 | 1.992x | 4.750x | 1.793x | -0.081x | 3.087 pp |

## Notes

This comparison isolates request count at the lowest successful KV-budget point. The row trial counts are shown explicitly so single-seed probes and replicated points can be distinguished.
