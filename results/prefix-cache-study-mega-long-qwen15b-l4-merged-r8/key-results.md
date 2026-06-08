# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.447% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 93.075% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 92.628 pp | 92.581 pp to 92.675 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.910 | -0.918 to -0.900 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 7.491 | 7.222 to 7.795 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.892 | -0.903 to -0.881 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.947 | -0.954 to -0.941 | Stable favorable decode TPOT effect; negative latency ratio is better. |
