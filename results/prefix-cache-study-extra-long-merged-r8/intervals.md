# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 94.404 pp
Direct counter delta               [|--------------------------------------------*--]  89.885 pp  89.867 pp to 89.909 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -0.881 to 1.101
p95 first-event/TTFT ratio delta   [----===*===----------|--------------------------]     -0.589  -0.694 to -0.445          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [---------------------|----------=======*======--]      0.761  0.489 to 1.011            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [---======*=====------|--------------------------]     -0.500  -0.747 to -0.284          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--=====*====---------|--------------------------]     -0.567  -0.791 to -0.399          Stable favorable decode TPOT effect; negative latency ratio is better.
```
