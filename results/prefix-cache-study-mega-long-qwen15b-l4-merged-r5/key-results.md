# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.447% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 93.070% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 92.623 pp | 92.549 pp to 92.698 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.908 | -0.921 to -0.894 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 7.487 | 7.057 to 7.921 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.898 | -0.913 to -0.881 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.949 | -0.958 to -0.939 | Stable favorable decode TPOT effect; negative latency ratio is better. |
