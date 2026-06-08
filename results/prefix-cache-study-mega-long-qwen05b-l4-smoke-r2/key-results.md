# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.446% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 93.051% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 92.605 pp | 92.512 pp to 92.697 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.911 | -0.952 to -0.870 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 3.969 | 3.261 to 4.678 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.842 | -0.869 to -0.814 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.902 | -0.938 to -0.865 | Stable favorable decode TPOT effect; negative latency ratio is better. |
