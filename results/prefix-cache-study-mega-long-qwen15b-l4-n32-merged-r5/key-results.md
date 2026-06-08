# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.446% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 96.238% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 95.792 pp | 95.715 pp to 95.869 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.941 | -0.962 to -0.915 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 9.711 | 8.764 to 10.618 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.922 | -0.937 to -0.904 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.735 | -0.786 to -0.686 | Stable favorable decode TPOT effect; negative latency ratio is better. |
