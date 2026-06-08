# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-mega-long-qwen15b-l4-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json`

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
scale: -1.391 to 8.232
p95 first-event/TTFT ratio delta   [--*----|----------------------------------------]     -0.910  -0.918 to -0.900          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [-------|----------------------------------=*==--]      7.491  7.222 to 7.795            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--*----|----------------------------------------]     -0.892  -0.903 to -0.881          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [--*----|----------------------------------------]     -0.947  -0.954 to -0.941          Stable favorable decode TPOT effect; negative latency ratio is better.
```
