# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-ultra-long-qwen05b-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 96.124 pp
Direct counter delta               [|--------------------------------------------*--]  91.460 pp  91.377 pp to 91.547 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.487 to 5.649
p95 first-event/TTFT ratio delta   [--=*=-----|-------------------------------------]     -0.964  -1.163 to -0.836          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [----------|-----------==========*=============--]      3.308  1.851 to 5.324            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [----=*=---|-------------------------------------]     -0.795  -0.947 to -0.652          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [----=*----|-------------------------------------]     -0.791  -0.857 to -0.709          Stable favorable decode TPOT effect; negative latency ratio is better.
```
