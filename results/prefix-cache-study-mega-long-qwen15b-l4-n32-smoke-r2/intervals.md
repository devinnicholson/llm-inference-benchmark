# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 100.662 pp
Direct counter delta               [|--------------------------------------------*--]  95.772 pp  95.677 pp to 95.868 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.514 to 10.354
p95 first-event/TTFT ratio delta   [--*---|-----------------------------------------]     -0.957  -0.974 to -0.940          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [------|-------------------------------------*=--]      9.717  9.619 to 9.815            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*---|-----------------------------------------]     -0.928  -0.946 to -0.910          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [---*--|-----------------------------------------]     -0.675  -0.677 to -0.672          Stable favorable decode TPOT effect; negative latency ratio is better.
```
