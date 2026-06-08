# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-ultra-long-qwen05b-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 96.160 pp
Direct counter delta               [|--------------------------------------------*--]  91.472 pp  91.372 pp to 91.581 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.589 to 3.184
p95 first-event/TTFT ratio delta   [--===*==--------|-------------------------------]     -1.032  -1.372 to -0.848          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [----------------|------------========*========--]      2.184  1.387 to 2.967            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [-----===*==-----|-------------------------------]     -0.809  -1.063 to -0.601          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [-------=*=------|-------------------------------]     -0.773  -0.857 to -0.629          Stable favorable decode TPOT effect; negative latency ratio is better.
```
