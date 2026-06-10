# Server Cache Pressure Curve

Source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n32-batched-tokens60640-seed3805-seed3906-cache-control-r1/server-cache-control-absolute.json`

| Phase | Profile | GPU Mem | n | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.325 | 32 | 8.080 | 0.442% | 0.975x | 1.026x | 1.026x | 1.038x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.325 | 32 | 8.091 | 96.204% | 10.741x | 0.095x | 0.085x | 0.313x |

## Notes

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Values near 1.0 are close to the reported KV capacity for the configured model length.
