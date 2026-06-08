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

## Next Step

The next GPU experiment should make prefill a larger fraction of total request
cost. Two reasonable directions are a larger model or longer shared prefixes.
The next software step is to keep improving the report around TTFT, direct cache
counters, and explicit non-claims for throughput and decode TPOT.
