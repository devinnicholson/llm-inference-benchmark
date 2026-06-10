# Server Cache Pressure Curve

Source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-n32-batched-tokens60640-seed3805-seed3906-cache-control-r1/server-cache-control-absolute.json`

| Phase | Profile | GPU Mem | n | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.350 | 32 | 3.298 | 0.438% | 0.995x | 1.006x | 1.008x | 1.032x |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.400 | 32 | 1.510 | 0.435% | 0.984x | 1.016x | 1.012x | 1.552x |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.450 | 32 | 0.979 | 0.432% | 0.995x | 1.005x | 1.006x | 1.252x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.350 | 32 | 3.303 | 96.204% | 8.467x | 0.119x | 0.105x | 0.091x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.400 | 32 | 1.512 | 96.204% | 8.638x | 0.116x | 0.100x | 0.176x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.450 | 32 | 0.981 | 96.204% | 9.125x | 0.109x | 0.095x | 0.046x |

## Notes

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Values near 1.0 are close to the reported KV capacity for the configured model length.
