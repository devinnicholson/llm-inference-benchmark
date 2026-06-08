# Prefix-Cache KV-Reuse Study

## Question

Can a controlled shared-prefix workload trigger measurable vLLM prefix-cache
reuse, and does that reuse show up in user-visible latency?

This study focuses on a narrow but interview-useful systems claim: not "prefix
caching always improves serving throughput", but "we can design a clean
workload/control pair, measure direct KV-cache reuse, and identify where that
reuse appears in the request lifecycle."

## Method

The current best result uses the no-repeat n=16 fixed-shape stability pass:

- Backend: vLLM on Modal
- GPU: NVIDIA T4
- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Execution: eager mode, prefix-cache metrics enabled
- Request count: `16`
- Output tokens: `8`
- Repeats: `8`
- Phase order: `cold_first`
- Warmup: one neutral long-prompt warmup per isolated call
- Shared profile: `shared_prefix_long_no_repeat_variant`
- Control profile: `matched_unique_prefix_no_repeat_variant`

The no-repeat prompt audit removed the duplicate-control confound. At n=16, both
profiles contain sixteen unique prompts and zero exact duplicate reusable block
tokens. The shared profile keeps a long reusable prompt prefix; the matched
control keeps similar prompt length but uses unique leading text.

## Primary Result

The preferred artifact is the TTFT-aware summary:

```text
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.md
```

The GitHub-facing key result table and interval chart are generated from that
summary:

```text
results/prefix-cache-study/key-results.md
results/prefix-cache-study/key-results.csv
results/prefix-cache-study/intervals.md
```

| Metric | Result |
| --- | ---: |
| Control direct counter hit rate | 4.716% |
| Shared direct counter hit rate | 83.138% |
| Shared-minus-control direct counter delta | 78.422 pp |
| Direct counter delta 90% bootstrap interval | 78.269 pp to 78.602 pp |
| p95 first-event/TTFT ratio delta | -0.284 |
| p95 first-event/TTFT 90% bootstrap interval | -0.497 to -0.073 |
| Throughput-ratio delta | -0.036 |
| Throughput-ratio 90% bootstrap interval | -0.332 to 0.185 |
| p95 latency-ratio delta | -0.074 |
| p95 latency-ratio 90% bootstrap interval | -0.180 to 0.023 |
| p95 stream TPOT-ratio delta | -0.036 |
| p95 stream TPOT-ratio 90% bootstrap interval | -0.113 to 0.044 |

Delta means shared-profile cache-to-cold ratio minus matched-control
cache-to-cold ratio. For latency-style ratios, negative is favorable: the
prefix-cache-enabled shared workload improved more than the matched control.

## Interpretation

The cache-reuse claim is strong. Direct measured-window counters show the shared
profile at `83.138%` cache hit rate while the matched control stays at `4.716%`.
The direct counter interval is tight and far from zero.

The timing claim is specific. The stable user-visible effect is p95
first-event/TTFT, with a 90% bootstrap interval from `-0.497` to `-0.073`.
That is the expected place for prefix caching to matter because reused KV blocks
reduce prefill work before the first generated token.

The study does not claim a stable throughput win. Throughput, end-to-end p95
latency, and stream TPOT intervals all cross zero in the eight-repeat pass. That
is a useful result, not a failure: it separates the cache mechanism from broader
serving throughput under a tiny model, short output length, and T4/eager setup.

## Extra-Long Follow-Up

Training 051 adds an extra-long no-repeat prompt/control pair that raises the
mean prompt length from roughly `327` tokens to roughly `1,185` tokens while
keeping the shared and control shapes matched.

The tokenizer audit shows:

- shared common-prefix full blocks: `72`
- control common-prefix full blocks: `1`
- shared-minus-control reusable block tokens: `17,040`
- exact duplicate reusable tokens: `0` for both profiles

The r3 GPU smoke was promising, and the merged r8 follow-up is stronger:

```text
results/prefix-cache-study-extra-long-merged-r8/key-results.md
results/prefix-cache-study-extra-long-merged-r8/intervals.md
```

In the r8 merge, direct counter reuse, p95 first-event/TTFT, throughput,
end-to-end p95 latency, and stream TPOT all stay on the favorable side of zero.
This is still a follow-up to the r8 no-repeat primary result, but it is now the
best evidence that larger prefill work can turn the cache mechanism into broader
serving wins under this setup.

Training 056 adds a Qwen2.5-0.5B model-shape smoke on the same extra-long
prompt pair, and Training 058 promotes it to an eight-repeat merged follow-up:

```text
results/prefix-cache-study-extra-long-qwen05b-merged-r8/key-results.md
results/prefix-cache-study-extra-long-qwen05b-merged-r8/intervals.md
```

The merged Qwen r8 result shows the same direct-cache mechanism survives a
larger small model on T4. Direct counter reuse, p95 first-event/TTFT,
throughput, end-to-end p95 latency, and stream TPOT all stay favorable in the
repeat-count-matched artifact, making it the best current model-shape
follow-up.

Training 059 adds an ultra-long Qwen smoke that raises the mean prompt length
to roughly `1,948` tokens and the shared common-prefix full blocks to `120`.
Training 061 extends it to an eight-repeat merged follow-up:

```text
results/prefix-cache-study-ultra-long-qwen05b-merged-r8/key-results.md
results/prefix-cache-study-ultra-long-qwen05b-merged-r8/intervals.md
```

The prompt audit is clean, and the r8 merge keeps every reported metric on the
favorable side of zero at the same repeat count as the extra-long Qwen follow-up:
direct counter delta `91.460 pp`, p95 first-event/TTFT ratio delta `-0.964`,
throughput-ratio delta `3.308`, p95 latency-ratio delta `-0.795`, and p95
stream TPOT-ratio delta `-0.791`. This is the strongest longer-context artifact
so far.

Training 062 adds a hardware-transfer smoke for the same ultra-long Qwen
workload on Modal L4:

```text
results/prefix-cache-study-ultra-long-qwen05b-l4-smoke-r2/key-results.md
results/prefix-cache-study-ultra-long-qwen05b-l4-smoke-r2/intervals.md
```

The L4 artifact records `NVIDIA L4, 23034` in `nvidia-smi` samples and reports
`772,020` GPU KV-cache tokens available for `max_model_len=2158`, versus roughly
`447,089` on the prior T4 runs. The r2 smoke preserves the direct-cache
mechanism with a `91.523 pp` direct counter delta and favorable timing
intervals. It is a transfer smoke, not yet a replacement for the T4 r8 result.

Training 063 uses the L4 path for a longer context shape instead of just
repeating the same ultra-long workload:

```text
results/prefix-cache-study-mega-long-qwen05b-l4-smoke-r2/key-results.md
results/prefix-cache-study-mega-long-qwen05b-l4-smoke-r2/intervals.md
```

The mega-long prompt audit raises the shared common prefix to `3,570` tokens,
or `223` full KV-cache blocks, while the matched unique-prefix control stays at
`1` full block. At n=16, both profiles still have sixteen unique prompts and
zero exact duplicate reusable tokens. On Modal L4, vLLM reported `695,249` GPU
KV-cache tokens available for `max_model_len=3790`, and the artifact records
`NVIDIA L4, 23034` in `nvidia-smi` samples. The r2 smoke showed a `92.605 pp`
direct counter delta and favorable timing intervals.

Training 064 promotes the mega-long L4 axis to r5:

```text
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r5/key-results.md
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r5/intervals.md
```

The merged r5 artifact has `10` scenario-level cold/cache paired runs, which
produce `5` shared/control comparisons. It keeps the direct-cache mechanism
stable with a `92.623 pp` direct counter delta and favorable timing intervals:
p95 first-event/TTFT ratio delta `-0.903`, throughput-ratio delta `3.908`, p95
latency-ratio delta `-0.829`, and p95 stream TPOT-ratio delta `-0.890`. This is
the strongest L4 context-shape artifact before the r8 promotion.

Training 065 promotes the mega-long L4 axis to r8, matching the repeat count of
the strongest T4 follow-ups:

```text
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r8/key-results.md
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r8/intervals.md
```

The merged r8 artifact has `16` scenario-level cold/cache paired runs, which
produce `8` shared/control comparisons. It keeps the direct-cache mechanism
stable with a `92.628 pp` direct counter delta and favorable timing intervals:
p95 first-event/TTFT ratio delta `-0.900`, throughput-ratio delta `3.903`, p95
latency-ratio delta `-0.825`, and p95 stream TPOT-ratio delta `-0.891`. This is
now the best L4 context-shape artifact and is repeat-count-matched to the T4
r8 results.

## Reproduce

Run the no-repeat n=16 fixed-shape stability pass:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long_no_repeat_variant,matched_unique_prefix_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 8 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8
```

Generate the TTFT-aware summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft
```

Build the GitHub-facing result table and interval chart:

```bash
python3 scripts/build_prefix_cache_study_table.py
```

## Artifact Map

Raw fixed-shape run:

```text
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-profile-control.csv
```

TTFT-aware summary:

```text
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.md
```

Generated report artifacts:

```text
results/prefix-cache-study/key-results.md
results/prefix-cache-study/key-results.csv
results/prefix-cache-study/intervals.md
```

Supporting prompt audit:

```text
results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16/prefix-cache-prompt-audit.md
```

Extra-long follow-up:

```text
results/modal-vllm-prefix-cache-prompt-audit-extra-long-no-repeat-n16/prefix-cache-prompt-audit.md
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-extra-long-merged-r8/key-results.md
results/prefix-cache-study-extra-long-merged-r8/intervals.md
results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-extra-long-qwen05b-smoke-r2/key-results.md
results/prefix-cache-study-extra-long-qwen05b-smoke-r2/intervals.md
results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-extra-long-qwen05b-merged-r5/key-results.md
results/prefix-cache-study-extra-long-qwen05b-merged-r5/intervals.md
results/modal-vllm-prefix-cache-extra-long-qwen05b-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-extra-long-qwen05b-merged-r8/key-results.md
results/prefix-cache-study-extra-long-qwen05b-merged-r8/intervals.md
results/modal-vllm-prefix-cache-prompt-audit-ultra-long-qwen05b-n16/prefix-cache-prompt-audit.md
results/modal-vllm-prefix-cache-ultra-long-qwen05b-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-ultra-long-qwen05b-smoke-r2/key-results.md
results/prefix-cache-study-ultra-long-qwen05b-smoke-r2/intervals.md
results/modal-vllm-prefix-cache-ultra-long-qwen05b-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-ultra-long-qwen05b-merged-r5/key-results.md
results/prefix-cache-study-ultra-long-qwen05b-merged-r5/intervals.md
results/modal-vllm-prefix-cache-ultra-long-qwen05b-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-ultra-long-qwen05b-merged-r8/key-results.md
results/prefix-cache-study-ultra-long-qwen05b-merged-r8/intervals.md
results/modal-vllm-prefix-cache-ultra-long-qwen05b-l4-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-ultra-long-qwen05b-l4-smoke-r2/key-results.md
results/prefix-cache-study-ultra-long-qwen05b-l4-smoke-r2/intervals.md
results/modal-vllm-prefix-cache-prompt-audit-mega-long-qwen05b-n16/prefix-cache-prompt-audit.md
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-smoke-r2/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-smoke-r2-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-mega-long-qwen05b-l4-smoke-r2/key-results.md
results/prefix-cache-study-mega-long-qwen05b-l4-smoke-r2/intervals.md
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-chunk-r3-seed1801/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-merged-r5/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-merged-r5-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r5/key-results.md
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r5/intervals.md
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-chunk-r3-seed1901/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-merged-r8/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-mega-long-qwen05b-l4-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r8/key-results.md
results/prefix-cache-study-mega-long-qwen05b-l4-merged-r8/intervals.md
```

## Next Step

Use the same L4 mega-long context shape to test a larger small model, or run a
backend comparison with the same prompt/control pair. The next useful artifact
should separate "longer reusable prefix" from "model size" or backend behavior.
