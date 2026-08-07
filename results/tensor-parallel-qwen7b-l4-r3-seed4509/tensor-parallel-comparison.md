# Tensor-Parallel Serving Comparison

Model: `Qwen/Qwen2.5-7B-Instruct` on one NVIDIA L4 versus 2 tensor-parallel L4s.

Matched workload: `shared_prefix_mega_long_no_repeat_variant`, n=16, output tokens=8, scheduler budget=8192, repeats=3.

## Capacity

| Topology | KV cache tokens | Max concurrency | P2P/custom all-reduce |
| --- | ---: | ---: | --- |
| 1x L4 / TP1 | 81784 | 22.49x | n/a |
| 2x L4 / TP2 | 445325 | 122.44x | unavailable; NCCL fallback |

## Request-path results

| Backend | Metric | TP1 median | TP2 median | TP2 / TP1 | 90% bootstrap interval |
| --- | --- | ---: | ---: | ---: | ---: |
| `server` | `throughput` | 7.726 | 9.650 | 1.242x | [1.239, 1.255] |
| `server` | `p95_ttft` | 16109.141 | 13011.762 | 0.813x | [0.804, 0.814] |
| `server` | `p95_latency` | 16561.541 | 13260.030 | 0.805x | [0.797, 0.807] |
| `server` | `p95_tpot` | 1944.929 | 1594.991 | 0.820x | [0.813, 0.827] |
| `async_llm` | `throughput` | 7.875 | 9.735 | 1.236x | [1.176, 1.283] |
| `async_llm` | `p95_ttft` | 15598.135 | 12793.802 | 0.820x | [0.817, 0.859] |
| `async_llm` | `p95_latency` | 16044.347 | 13041.784 | 0.813x | [0.783, 0.852] |
| `async_llm` | `p95_tpot` | 2059.505 | 1580.186 | 0.767x | [0.765, 0.791] |

For throughput, values above 1.0 favor TP2. For TTFT, latency, and TPOT, values below 1.0 favor TP2.

## Interpretation boundary

This is a controlled topology experiment, not a production-scale claim. The bootstrap interval resamples matched repeat indices and does not capture variation across hosts, regions, or GPU interconnects.
