# Tensor-Parallel Serving Comparison

Model: `Qwen/Qwen2.5-7B-Instruct` on one NVIDIA L4 versus 2 tensor-parallel L4s.

Matched workload: `shared_prefix_mega_long_no_repeat_variant`, n=16, output tokens=8, scheduler budget=8192, repeats=3.

## Capacity

| Topology | KV cache tokens | Max concurrency | P2P/custom all-reduce |
| --- | ---: | ---: | --- |
| 1x L4 / TP1 | 81785 | 22.39x | n/a |
| 2x L4 / TP2 | 445331 | 121.91x | unavailable; NCCL fallback |

## Request-path results

| Backend | Metric | TP1 median | TP2 median | TP2 / TP1 | 90% bootstrap interval |
| --- | --- | ---: | ---: | ---: | ---: |
| `server` | `throughput` | 8.195 | 11.103 | 1.356x | [1.333, 1.365] |
| `server` | `p95_ttft` | 15161.300 | 11276.046 | 0.743x | [0.738, 0.756] |
| `server` | `p95_latency` | 15615.836 | 11523.049 | 0.737x | [0.732, 0.750] |
| `server` | `p95_tpot` | 1834.954 | 1368.216 | 0.746x | [0.739, 0.763] |
| `async_llm` | `throughput` | 8.304 | 11.254 | 1.354x | [1.269, 1.397] |
| `async_llm` | `p95_ttft` | 14845.749 | 11013.472 | 0.742x | [0.740, 0.792] |
| `async_llm` | `p95_latency` | 15282.491 | 11260.202 | 0.737x | [0.716, 0.786] |
| `async_llm` | `p95_tpot` | 1999.992 | 1352.098 | 0.693x | [0.675, 0.728] |

For throughput, values above 1.0 favor TP2. For TTFT, latency, and TPOT, values below 1.0 favor TP2.

## Interpretation boundary

This is a controlled topology experiment, not a production-scale claim. The bootstrap interval resamples matched repeat indices and does not capture variation across hosts, regions, or GPU interconnects.
