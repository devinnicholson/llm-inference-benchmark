# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-batched-tokens60640-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 100.663 pp
Direct counter delta               [|--------------------------------------------*--]  95.830 pp  95.754 pp to 95.869 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.487 to 10.178
p95 first-event/TTFT ratio delta   [--*---|-----------------------------------------]     -0.949  -0.957 to -0.940          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [------|--------------------------=======*=====--]      8.343  6.779 to 9.648            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*---|-----------------------------------------]     -0.909  -0.923 to -0.892          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--*---|-----------------------------------------]     -0.931  -0.955 to -0.904          Stable favorable decode TPOT effect; negative latency ratio is better.
```
