# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n32-batched-tokens60640-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 100.663 pp
Direct counter delta               [|--------------------------------------------*--]  95.869 pp  95.868 pp to 95.870 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.539 to 11.057
p95 first-event/TTFT ratio delta   [--*---|-----------------------------------------]     -0.961  -0.966 to -0.956          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [------|----------------------------------==*==--]     10.029  9.572 to 10.485           Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*---|-----------------------------------------]     -0.922  -0.930 to -0.914          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--*---|-----------------------------------------]     -0.955  -0.963 to -0.947          Stable favorable decode TPOT effect; negative latency ratio is better.
```
