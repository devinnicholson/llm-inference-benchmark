# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-ultra-long-qwen05b-l4-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 96.200 pp
Direct counter delta               [|--------------------------------------------*--]  91.523 pp  91.426 pp to 91.619 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.272 to 5.485
p95 first-event/TTFT ratio delta   [--*=-----|--------------------------------------]     -0.925  -0.965 to -0.886          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [---------|--------==============*=============--]      3.268  1.358 to 5.178            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [---=*=---|--------------------------------------]     -0.738  -0.855 to -0.620          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [---*=----|--------------------------------------]     -0.795  -0.841 to -0.749          Stable favorable decode TPOT effect; negative latency ratio is better.
```
