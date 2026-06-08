# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 97.333 pp
Direct counter delta               [|--------------------------------------------*--]  92.623 pp  92.549 pp to 92.698 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.402 to 8.365
p95 first-event/TTFT ratio delta   [--*----|----------------------------------------]     -0.908  -0.921 to -0.894          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [-------|---------------------------------==*==--]      7.487  7.057 to 7.921            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*=---|----------------------------------------]     -0.898  -0.913 to -0.881          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--*----|----------------------------------------]     -0.949  -0.958 to -0.939          Stable favorable decode TPOT effect; negative latency ratio is better.
```
