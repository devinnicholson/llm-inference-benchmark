# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-smoke-r3-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 94.418 pp
Direct counter delta               [|--------------------------------------------*--]  89.891 pp  89.860 pp to 89.922 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.220 to 1.103
p95 first-event/TTFT ratio delta   [-----------=*=-----------|----------------------]     -0.618  -0.671 to -0.565          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [-------------------------|------------====*===--]      0.834  0.671 to 0.998            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--=======*========-------|----------------------]     -0.754  -1.107 to -0.401          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--=======*======---------|----------------------]     -0.800  -1.114 to -0.485          Stable favorable decode TPOT effect; negative latency ratio is better.
```
