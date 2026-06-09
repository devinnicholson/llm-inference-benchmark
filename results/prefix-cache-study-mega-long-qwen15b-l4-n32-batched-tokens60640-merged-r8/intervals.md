# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-batched-tokens60640-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 100.662 pp
Direct counter delta               [|--------------------------------------------*--]  95.821 pp  95.772 pp to 95.869 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.518 to 10.754
p95 first-event/TTFT ratio delta   [--*---|-----------------------------------------]     -0.954  -0.960 to -0.947          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [------|-----------------------------=====*====--]      9.184  7.959 to 10.196           Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*---|-----------------------------------------]     -0.916  -0.926 to -0.902          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--*---|-----------------------------------------]     -0.937  -0.954 to -0.918          Stable favorable decode TPOT effect; negative latency ratio is better.
```
