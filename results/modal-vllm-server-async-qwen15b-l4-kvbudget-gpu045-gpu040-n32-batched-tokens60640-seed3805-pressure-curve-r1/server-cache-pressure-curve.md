# Server Cache Pressure Curve

Source: `results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-n32-batched-tokens60640-seed3805-cache-control-r1/server-cache-control-absolute.json`

| Phase | Profile | GPU Mem | n | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.400 | 32 | 1.510 | 0.436% | 0.963x | 1.038x | 1.029x | 1.002x |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 0.450 | 32 | 0.979 | 0.433% | 0.987x | 1.013x | 1.014x | 0.883x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.400 | 32 | 1.509 | 96.109% | 8.100x | 0.123x | 0.110x | 0.111x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 0.450 | 32 | 0.978 | 96.109% | 9.010x | 0.111x | 0.093x | 0.071x |

## Notes

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Values near 1.0 are close to the reported KV capacity for the configured model length.
