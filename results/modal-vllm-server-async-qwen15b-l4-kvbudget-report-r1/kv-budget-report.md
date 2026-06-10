# KV-Budget Report

Pressure curve source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-gpu0325-n32-batched-tokens60640-seed3805-seed3906-pressure-curve-r1/server-cache-pressure-curve.json`
Startup floor source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu030-n32-batched-tokens60640-feasibility-failure-r1/failure.json`

## Headline

`gpu_memory_utilization=0.300` fails during vLLM KV-cache initialization, while `gpu_memory_utilization=0.325` runs successfully.

At the lowest successful budget, the shared-prefix workload reaches 8.091x estimated prompt pressure with 96.204% cache hit rate, 10.741x throughput, and 0.095x p95 latency relative to cache-off.

Matched-unique control throughput stays near neutral across the successful curve: 0.975x to 0.995x.

## Startup Floor

| GPU Mem | Status | Phase | Available KV Memory | Interpretation |
| ---: | --- | --- | ---: | --- |
| 0.300 | `failed_before_artifact_write` | AsyncLLM engine KV-cache initialization | -0.170 GiB | For this Qwen2.5-1.5B L4 n32 workload shape, gpu_memory_utilization=0.30 is below the vLLM startup floor before any measured cache-control comparison can run. |
| 0.325 | `measured` | paired server/Async cache-control benchmark | n/a | Lowest successful budget point in the replicated curve. |

## Successful Budget Curve

| Profile | GPU Mem | KV Tokens | Max Concurrency | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | Trials |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique` | 0.325 | 14,206 | 3.900 | 8.080 | 0.442% | 0.975x | 1.026x | 2 |
| `matched_unique` | 0.350 | 34,805 | 9.550 | 3.298 | 0.438% | 0.995x | 1.006x | 2 |
| `matched_unique` | 0.400 | 76,018 | 20.860 | 1.510 | 0.435% | 0.984x | 1.016x | 2 |
| `matched_unique` | 0.450 | 117,214 | 32.170 | 0.979 | 0.432% | 0.995x | 1.005x | 2 |
| `shared_prefix` | 0.325 | 14,194 | 3.910 | 8.091 | 96.204% | 10.741x | 0.095x | 2 |
| `shared_prefix` | 0.350 | 34,776 | 9.570 | 3.303 | 96.204% | 8.467x | 0.119x | 2 |
| `shared_prefix` | 0.400 | 75,955 | 20.910 | 1.512 | 96.204% | 8.638x | 0.116x | 2 |
| `shared_prefix` | 0.450 | 117,118 | 32.240 | 0.981 | 96.204% | 9.125x | 0.109x | 2 |

## Reading The Table

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Throughput and latency ratios are cache-on divided by cache-off for the same prompt profile, request count, and budget.
