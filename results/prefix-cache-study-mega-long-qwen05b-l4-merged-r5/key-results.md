# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 0.447% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 93.070% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 92.623 pp | 92.549 pp to 92.698 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.903 | -0.923 to -0.884 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 3.908 | 3.487 to 4.297 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.829 | -0.845 to -0.812 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.890 | -0.910 to -0.872 | Stable favorable decode TPOT effect; negative latency ratio is better. |
