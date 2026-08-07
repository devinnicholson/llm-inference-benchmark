# Replicated Tensor-Parallel Serving Study

`Qwen/Qwen2.5-7B-Instruct` on one NVIDIA L4 versus two tensor-parallel L4s, with cache disabled on both execution paths.

Protocol: `shared_prefix_mega_long_no_repeat_variant`, n=16, output tokens=8, scheduler budget=8192. The aggregate contains 2 independent run/seed trials and 6 paired repeats.

## Capacity by independent trial

| Seed | TP1 KV tokens | TP2 KV tokens | TP2 / TP1 | TP2 interconnect |
| ---: | ---: | ---: | ---: | --- |
| 4409 | 81785 | 445331 | 5.445x | NCCL fallback; no P2P/custom all-reduce |
| 4509 | 81784 | 445325 | 5.445x | NCCL fallback; no P2P/custom all-reduce |

## Replicated request-path results

| Backend | Metric | TP1 median | TP2 median | TP2 / TP1 | Hierarchical 90% interval | Trial-median range |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `server` | `throughput` | 7.952 | 10.307 | 1.294x | [1.240, 1.361] | [1.242, 1.356] |
| `server` | `p95_ttft` | 15645.021 | 12218.105 | 0.780x | [0.741, 0.813] | [0.743, 0.813] |
| `server` | `p95_latency` | 16098.742 | 12464.965 | 0.774x | [0.735, 0.806] | [0.737, 0.805] |
| `server` | `p95_tpot` | 1895.159 | 1489.959 | 0.788x | [0.743, 0.824] | [0.746, 0.820] |
| `async_llm` | `throughput` | 7.977 | 10.188 | 1.276x | [1.206, 1.376] | [1.236, 1.354] |
| `async_llm` | `p95_ttft` | 15176.083 | 12206.139 | 0.805x | [0.741, 0.840] | [0.742, 0.820] |
| `async_llm` | `p95_latency` | 15880.409 | 12455.005 | 0.784x | [0.726, 0.832] | [0.737, 0.813] |
| `async_llm` | `p95_tpot` | 2024.025 | 1511.389 | 0.747x | [0.684, 0.779] | [0.693, 0.767] |

Throughput ratios above 1.0 favor TP2; latency-style ratios below 1.0 favor TP2.

## Interpretation boundary

The hierarchical bootstrap resamples independent Modal run/seed trials and then repeats within each trial. The artifacts do not expose physical host identity. Two trials check whether the effect replicates, but they do not characterize the full Modal L4 fleet or other interconnect topologies. This remains a controlled serving study, not a production-traffic benchmark.
