# Server Cache Pressure Curve

Source: `results/modal-vllm-server-async-qwen15b-l4-pressure-edge-n42-n44-batched-tokens60640-seed3704-cache-control-r1/server-cache-control-absolute.json`

| Phase | Profile | n | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 42 | 0.951 | 0.436% | 1.063x | 0.940x | 1.000x | 0.936x |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 44 | 0.997 | 0.436% | 1.042x | 0.960x | 0.961x | 1.110x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 42 | 0.959 | 97.113% | 10.969x | 0.091x | 0.079x | 0.038x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 44 | 1.004 | 97.231% | 7.307x | 0.137x | 0.130x | 0.050x |

## Notes

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Values near 1.0 are close to the reported KV capacity for the configured model length.
