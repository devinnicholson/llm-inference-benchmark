# Server Cache Pressure Curve

Source: `results/modal-vllm-server-async-qwen15b-l4-pressure-curve-n16-n40-batched-tokens60640-seed3603-cache-control-r1/server-cache-control-absolute.json`

| Phase | Profile | n | Prompt Pressure | Hit Rate | Throughput Ratio | p95 Latency Ratio | TTFT Ratio | TPOT Ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 16 | 0.362 | 0.418% | 1.056x | 0.947x | 0.969x | 0.524x |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | 40 | 0.906 | 0.434% | 1.032x | 0.969x | 0.972x | 1.039x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 16 | 0.363 | 93.116% | 6.278x | 0.159x | 0.127x | 0.984x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | 40 | 0.909 | 96.984% | 8.447x | 0.118x | 0.102x | 0.029x |

## Notes

Prompt pressure is estimated prompt tokens divided by the parsed server GPU KV-cache token capacity. Values near 1.0 are close to the reported KV capacity for the configured model length.
