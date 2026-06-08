# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 82.532 pp
Direct counter delta               [|--------------------------------------------*--]  78.422 pp  78.269 pp to 78.602 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -0.532 to 0.220
p95 first-event/TTFT ratio delta   [--=============*==============---|--------------]     -0.284  -0.497 to -0.073          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [------------===================*=|============--]     -0.036  -0.332 to 0.185           Interval crosses zero; no stable throughput claim.
p95 latency-ratio delta            [----------------------=======*===|==------------]     -0.074  -0.180 to 0.023           Interval crosses zero; no stable end-to-end latency claim.
p95 stream TPOT-ratio delta        [--------------------------=====*=|===-----------]     -0.036  -0.113 to 0.044           Interval crosses zero; no stable decode TPOT claim.
```
