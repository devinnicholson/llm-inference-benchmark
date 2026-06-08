# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 97.332 pp
Direct counter delta               [|--------------------------------------------*--]  92.605 pp  92.512 pp to 92.697 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.363 to 7.767
p95 first-event/TTFT ratio delta   [--*----|----------------------------------------]     -0.915  -0.915 to -0.915          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [-------|-------------------------------------*--]      7.342  7.333 to 7.352            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*----|----------------------------------------]     -0.898  -0.908 to -0.888          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--*----|----------------------------------------]     -0.948  -0.948 to -0.947          Stable favorable decode TPOT effect; negative latency ratio is better.
```
