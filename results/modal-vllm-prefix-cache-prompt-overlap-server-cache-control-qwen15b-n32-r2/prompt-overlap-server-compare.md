# Prompt Overlap Vs Server Cache-Control

## Summary

| Metric | Value |
| --- | ---: |
| Audit trials | 2 |
| Shared common-prefix full blocks | 222.500 |
| Control common-prefix full blocks | 1.000 |
| Shared-minus-control full-block delta | 221.500 |
| Shared-minus-control reusable block-token delta | 109864.000 |
| Control exact-duplicate reusable block tokens | 0.000 |
| Matched-unique min server cache-on/off throughput ratio | 10.821x |
| Matched-unique max server cache-on/off p95 latency ratio | 0.096x |

## Joined Server View

| Phase Order | Profile | Overlap Class | Common Blocks | Server Throughput Ratio | Server p95 Latency Ratio |
| --- | --- | --- | ---: | ---: | ---: |
| `async_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | `minimal_leading_overlap` | 1.000 | 10.821x | 0.096x |
| `async_first` | `shared_prefix_mega_long_no_repeat_variant` | `large_leading_overlap` | 222.500 | 11.308x | 0.088x |
| `server_first` | `matched_unique_prefix_mega_long_no_repeat_variant` | `minimal_leading_overlap` | 1.000 | 12.091x | 0.083x |
| `server_first` | `shared_prefix_mega_long_no_repeat_variant` | `large_leading_overlap` | 222.500 | 13.841x | 0.073x |

## Interpretation

The prompt-token audit says the matched-unique control has only one common leading full block and no exact duplicate prompts. That is not enough leading-token reuse to explain the order-of-magnitude server cache-on/cache-off speedup by the intended shared-prefix mechanism alone.

The next measurement should separate explicit user-visible shared prefixes from other server-path effects: vLLM prefix-cache semantics, scenario ordering, shared chat-template scaffolding, or generic cache-on server behavior under this workload shape.
