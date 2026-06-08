# Prefix-Cache Study Interval Chart

Source: `results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.json`

`*` marks the mean effect, `=` marks the 90% bootstrap interval, and `|` marks zero.

## Direct Cache Counter Delta

Positive percentage points mean the shared-prefix profile reused more KV cache.

```text
scale: 0.000 pp to 95.006 pp
Direct counter delta               [|-------------------------------------------=*--]  90.099 pp  89.716 pp to 90.482 pp    Stable positive reuse effect; interval is far above zero.
```

## Cache-To-Cold Ratio Deltas

For latency-style ratios, negative is favorable. Intervals crossing zero are non-claims.

```text
scale: -1.488 to 2.971
p95 first-event/TTFT ratio delta   [------=*--------|-------------------------------]     -0.838  -0.896 to -0.779          Stable favorable first-token effect; negative latency ratio is better.
Throughput-ratio delta             [----------------|-------------------====*=====--]      2.341  1.913 to 2.768            Stable favorable throughput effect; positive ratio is better.
p95 latency-ratio delta            [--====*====-----|-------------------------------]     -0.934  -1.285 to -0.582          Stable favorable end-to-end latency effect; negative latency ratio is better.
p95 stream TPOT-ratio delta        [-------=*-------|-------------------------------]     -0.742  -0.801 to -0.684          Stable favorable decode TPOT effect; negative latency ratio is better.
```
