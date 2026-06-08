# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 100.662 pp
Direct counter delta               [|--------------------------------------------*--]  95.792 pp  95.715 pp to 95.869 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.541 to 11.197
p95 first-event/TTFT ratio delta   [--*---|-----------------------------------------]     -0.941  -0.962 to -0.915          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [------|-------------------------------====*===--]      9.711  8.764 to 10.618           Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*---|-----------------------------------------]     -0.922  -0.937 to -0.904          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [---*--|-----------------------------------------]     -0.735  -0.786 to -0.686          Stable favorable decode TPOT effect; negative latency ratio is better.
```
