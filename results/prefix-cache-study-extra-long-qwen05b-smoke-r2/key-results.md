# Prefix-Cache Study Key Results

Source: `results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

| Metric | Value | 90% bootstrap interval | Interpretation |
| --- | ---: | --- | --- |
| Control direct counter hit rate | 1.419% | n/a | Matched unique-prefix baseline stays low after the no-repeat prompt audit. |
| Shared direct counter hit rate | 91.518% | n/a | Shared-prefix workload triggers measured-window KV reuse. |
| Shared-minus-control direct counter delta | 90.099 pp | 89.716 pp to 90.482 pp | Stable positive reuse effect; interval is far above zero. |
| p95 first-event/TTFT ratio delta | -0.838 | -0.896 to -0.779 | Stable favorable first-token effect; negative latency ratio is better. |
| Throughput-ratio delta | 2.341 | 1.913 to 2.768 | Stable favorable throughput effect; positive ratio is better. |
| p95 latency-ratio delta | -0.934 | -1.285 to -0.582 | Stable favorable end-to-end latency effect; negative latency ratio is better. |
| p95 stream TPOT-ratio delta | -0.742 | -0.801 to -0.684 | Stable favorable decode TPOT effect; negative latency ratio is better. |
