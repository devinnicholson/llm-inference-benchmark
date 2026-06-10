# Server Cache Pressure Curve

Source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n16-batched-tokens60640-seed4107-cache-control-r1/server-cache-control-absolute.json`

| Phase | Profile | GPU Mem | n | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.325 | 16 | 4.036 | 0.438% | 0.938x | 1.066x | 1.061x | 1.064x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.325 | 16 | 4.053 | 93.116% | 4.619x | 0.216x | 0.161x | 0.627x |

## Notes

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Values near 1.0 are close to the reported KV capacity for the configured model length.
