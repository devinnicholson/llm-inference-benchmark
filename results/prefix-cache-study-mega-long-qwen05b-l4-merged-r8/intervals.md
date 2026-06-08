# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 97.309 pp
Direct counter delta               [|--------------------------------------------*--]  92.628 pp  92.581 pp to 92.675 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.169 to 4.426
p95 first-event/TTFT ratio delta   [--*-------|-------------------------------------]     -0.900  -0.915 to -0.887          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [----------|-----------------------------===*==--]      3.903  3.644 to 4.171            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [---*------|-------------------------------------]     -0.825  -0.837 to -0.814          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--*-------|-------------------------------------]     -0.891  -0.906 to -0.878          Stable favorable decode TPOT effect; negative latency ratio is better.
```
