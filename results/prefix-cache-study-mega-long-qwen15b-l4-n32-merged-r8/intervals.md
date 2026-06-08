# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 100.637 pp
Direct counter delta               [|--------------------------------------------*--]  95.797 pp  95.748 pp to 95.845 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.540 to 11.295
p95 first-event/TTFT ratio delta   [--*---|-----------------------------------------]     -0.942  -0.957 to -0.924          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [------|--------------------------------===*===--]      9.863  9.015 to 10.711           Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*---|-----------------------------------------]     -0.911  -0.925 to -0.897          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [---*--|-----------------------------------------]     -0.683  -0.736 to -0.632          Stable favorable decode TPOT effect; negative latency ratio is better.
```
