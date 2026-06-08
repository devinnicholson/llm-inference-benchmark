# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 1.342% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 91.233% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 89.891 pp | 89.867 pp to 89.915 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.561 | -0.701 to -0.376 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 0.772 | 0.411 to 1.111 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.525 | -0.867 to -0.235 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.583 | -0.888 to -0.356 | Stable favorable decode TPOT effect; negative latency ratio is better. |
