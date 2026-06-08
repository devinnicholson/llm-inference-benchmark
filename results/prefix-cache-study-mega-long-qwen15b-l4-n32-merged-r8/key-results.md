# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.446% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 96.243% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 95.797 pp | 95.748 pp to 95.845 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.942 | -0.957 to -0.924 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 9.863 | 9.015 to 10.711 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.911 | -0.925 to -0.897 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.683 | -0.736 to -0.632 | Stable favorable decode TPOT effect; negative latency ratio is better. |
