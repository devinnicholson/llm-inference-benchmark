# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 94.865 pp
Direct counter delta               [|--------------------------------------------*--]  90.157 pp  89.965 pp to 90.348 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -0.940 to 2.027
p95 first-event/TTFT ratio delta   [--=*=----------|--------------------------------]     -0.757  -0.805 to -0.705          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [---------------|------------------=====*======--]      1.541  1.199 to 1.892            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [---==*===------|--------------------------------]     -0.597  -0.776 to -0.440          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [---=*=---------|--------------------------------]     -0.695  -0.739 to -0.638          Stable favorable decode TPOT effect; negative latency ratio is better.
```
