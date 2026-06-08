# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 94.942 pp
Direct counter delta               [|-------------------------------------------=*--]  90.145 pp  89.869 pp to 90.421 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.065 to 2.365
p95 first-event/TTFT ratio delta   [---*=----------|--------------------------------]     -0.811  -0.848 to -0.777          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [---------------|----------------=======*======--]      1.747  1.276 to 2.210            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--====*===-----|--------------------------------]     -0.633  -0.909 to -0.392          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [----=*=--------|--------------------------------]     -0.692  -0.763 to -0.603          Stable favorable decode TPOT effect; negative latency ratio is better.
```
