# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.447% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 93.075% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 92.628 pp | 92.581 pp to 92.675 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.900 | -0.915 to -0.887 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 3.903 | 3.644 to 4.171 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.825 | -0.837 to -0.814 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.891 | -0.906 to -0.878 | Stable favorable decode TPOT effect; negative latency ratio is better. |
