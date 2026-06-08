# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-ultra-long-qwen05b-l4-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.822% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 92.344% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 91.523 pp | 91.426 pp to 91.619 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.925 | -0.965 to -0.886 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 3.268 | 1.358 to 5.178 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.738 | -0.855 to -0.620 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.795 | -0.841 to -0.749 | Stable favorable decode TPOT effect; negative latency ratio is better. |
