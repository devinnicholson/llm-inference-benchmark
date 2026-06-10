# Server Cache Pressure Curve

Source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n32-batched-tokens60640-seed3906-cache-control-r1/server-cache-control-absolute.json`

| Phase | Profile | GPU Mem | n | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.325 | 32 | 8.080 | 0.442% | 0.968x | 1.034x | 1.033x | 1.049x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.325 | 32 | 8.109 | 96.300% | 12.456x | 0.080x | 0.071x | 0.237x |

## Notes

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Values near 1.0 are close to the reported KV capacity for the configured model length.
