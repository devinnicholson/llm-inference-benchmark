# llm-inference-benchmark-lab

Research-grade learning repo for **568 Systems and Machine Learning**.

The project is now centered on **KV-cache behavior in LLM serving**. The goal is
to build an ML inference systems artifact that can survive an ML infra interview:
clear workload definitions, request lifecycle traces, benchmark methodology,
scheduler experiments, KV-cache pressure studies, and eventually backend
comparisons against real inference engines.

Current report: [`docs/prefix-cache-study.md`](docs/prefix-cache-study.md)
summarizes the no-repeat vLLM prefix-cache result, longer-context follow-ups,
and the first L4-scale context smoke. The primary no-repeat claim is stable
measured-window KV-cache reuse plus a p95 first-event/TTFT improvement; the
longer Qwen follow-ups now show broader favorable timing when prefill work is
larger.

Generated result artifacts:
[`results/prefix-cache-study/key-results.md`](results/prefix-cache-study/key-results.md)
and [`results/prefix-cache-study/intervals.md`](results/prefix-cache-study/intervals.md)

Latest extra-long follow-up:
[`results/prefix-cache-study-extra-long-merged-r8/key-results.md`](results/prefix-cache-study-extra-long-merged-r8/key-results.md)

Latest model-shape follow-up:
[`results/prefix-cache-study-extra-long-qwen05b-merged-r8/key-results.md`](results/prefix-cache-study-extra-long-qwen05b-merged-r8/key-results.md)

Latest longer-context follow-up:
[`results/prefix-cache-study-ultra-long-qwen05b-merged-r8/key-results.md`](results/prefix-cache-study-ultra-long-qwen05b-merged-r8/key-results.md)

Latest L4-scale context follow-up:
[`results/prefix-cache-study-mega-long-qwen05b-l4-merged-r8/key-results.md`](results/prefix-cache-study-mega-long-qwen05b-l4-merged-r8/key-results.md)

## Course Track

This repo starts with the 568 path:

1. Request lifecycle, KV-cache accounting, and measurement vocabulary
2. Profiling and trace discipline
3. Triton primitives for inference hot paths
4. KV cache, batching, and scheduling policies
5. Quantization and decoding tradeoffs
6. Serving APIs, streaming, cancellation, and metrics
7. Distributed inference and placement policies
8. Trace-driven workload realism
9. Artifact report and reproducibility package

## Week 1 Artifact

Week 1 defines the basic serving lifecycle before any real model backend exists.
The initial code provides:

- A workload schema for inference requests
- Validation for prompt/output token counts and arrival times
- A deterministic FIFO request lifecycle simulator
- Per-stage traces for queueing, tokenization, prefill, decode, and streaming
- Per-request KV-cache footprint estimates
- Active KV-cache timeline metrics
- Summary metrics for end-to-end latency, queue wait, throughput, and memory
  pressure

Run the starter workload:

```bash
python3 scripts/replay_workload.py workloads/week01_mixed_requests.json \
  --model-config configs/models/llama-7b-gqa-fp16.json
```

Generate a deterministic bursty workload:

```bash
python3 scripts/generate_workload.py mixed_bursty \
  --requests 32 \
  --seed 568 \
  --output workloads/generated/mixed_bursty_32_seed568.json
```

Replay it with overlapping FIFO request slots:

```bash
python3 scripts/replay_workload.py workloads/generated/mixed_bursty_32_seed568.json \
  --model-config configs/models/llama-7b-gqa-fp16.json \
  --max-concurrent-requests 4 \
  --scheduler-policy fifo
```

Compare a scheduler policy:

```bash
python3 scripts/replay_workload.py workloads/generated/mixed_bursty_32_seed568.json \
  --model-config configs/models/llama-7b-gqa-fp16.json \
  --max-concurrent-requests 4 \
  --scheduler-policy shortest-cache
```

Replay with a constrained KV-cache budget:

```bash
python3 scripts/replay_workload.py workloads/generated/mixed_bursty_32_seed568.json \
  --model-config configs/models/llama-7b-gqa-fp16.json \
  --capacity-config configs/capacity/tight-1gb-kv.json \
  --max-concurrent-requests 4 \
  --scheduler-policy memory-aware-deadline
```

Run the first reproducible capacity sweep:

```bash
python3 scripts/run_sweep.py
```

The sweep writes JSON and CSV results to
[`results/experiment-001-capacity-sweep`](results/experiment-001-capacity-sweep)
and is documented in
[`docs/experiment-001-capacity-sweep.md`](docs/experiment-001-capacity-sweep.md).

Run the first Modal remote sweep:

```bash
modal run modal_app.py
```

Run the first Modal GPU probe:

```bash
modal run modal_app.py --mode gpu-probe
```

Run the first Modal tiny inference timing:

```bash
modal run modal_app.py --mode tiny-inference
```

Run the first Modal vLLM inference baseline:

```bash
modal run modal_app.py --mode vllm-inference
```

Inspect vLLM cache-metrics support in the Modal image:

```bash
modal run modal_app.py --mode vllm-cache-metrics-probe
```

Run a tiny GPU-backed vLLM cache-metrics smoke:

```bash
modal run modal_app.py --mode vllm-cache-metrics-smoke \
  --prompt-profile shared_prefix_long \
  --prompt-count 4 \
  --max-new-tokens 8
```

Run the first Modal vLLM streaming timing:

```bash
modal run modal_app.py --mode vllm-streaming
```

Run the first OpenAI-compatible vLLM server streaming smoke:

```bash
modal run modal_app.py --mode vllm-server-streaming
```

Run the first OpenAI-compatible vLLM server concurrent streaming workload:

```bash
modal run modal_app.py --mode vllm-server-concurrent
```

Run the repeated OpenAI-compatible vLLM server sweep:

```bash
modal run modal_app.py --mode vllm-server-sweep \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1
```

Compare the repeated server sweep against the in-process `AsyncLLM` sweep:

```bash
modal run modal_app.py --mode vllm-server-sweep-compare
```

Run a paired same-worker server-vs-`AsyncLLM` benchmark:

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1
```

Run the server-first phase-order control:

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order server_first \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1
```

Compare paired phase orders:

```bash
modal run modal_app.py --mode vllm-server-async-phase-order-compare
```

Aggregate multiple phase-order comparisons:

```bash
modal run modal_app.py --mode vllm-server-async-multitrial-aggregate
```

Run the long-prompt phase-order workload:

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order async_first \
  --prompt-profiles long \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --output-dir results/modal-vllm-server-async-paired-long-async-first

modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order server_first \
  --prompt-profiles long \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --output-dir results/modal-vllm-server-async-paired-long-server-first
```

Compare the long-prompt phase orders and then summarize short-vs-long workload sensitivity:

```bash
modal run modal_app.py --mode vllm-server-async-phase-order-compare \
  --async-first-paired-dir results/modal-vllm-server-async-paired-long-async-first \
  --server-first-paired-dir results/modal-vllm-server-async-paired-long-server-first \
  --output-dir results/modal-vllm-server-async-phase-order-compare-long

modal run modal_app.py --mode vllm-server-async-workload-compare
```

Run the first Modal vLLM concurrent workload:

```bash
modal run modal_app.py --mode vllm-concurrent
```

Run the first Modal vLLM concurrency/context sweep:

```bash
modal run modal_app.py --mode vllm-sweep
```

Run the repeated sweep with an explicit shuffle seed:

```bash
modal run modal_app.py --mode vllm-sweep --repeats 3 --scenario-seed 568
```

Run the paired prefix-cache sweep and comparison:

```bash
modal run modal_app.py --mode vllm-sweep --prefix-caching on
modal run modal_app.py --mode vllm-prefix-cache-compare
```

Run the shared-prefix KV-cache pilot:

```bash
modal run modal_app.py --mode vllm-sweep \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --scenario-seed 568 \
  --prefix-caching off \
  --output-dir results/modal-vllm-shared-prefix-cold

modal run modal_app.py --mode vllm-sweep \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --scenario-seed 568 \
  --prefix-caching on \
  --output-dir results/modal-vllm-shared-prefix-cache

modal run modal_app.py --mode vllm-prefix-cache-compare \
  --cold-sweep-dir results/modal-vllm-shared-prefix-cold \
  --prefix-sweep-dir results/modal-vllm-shared-prefix-cache \
  --output-dir results/modal-vllm-shared-prefix-cache-compare
```

Run the paired same-worker prefix-cache control:

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --phase-order cold_first

modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --phase-order cache_first

modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare
```

Run the long shared-prefix versus matched unique-prefix control:

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --phase-order cold_first \
  --output-dir results/modal-vllm-prefix-cache-long-control-paired

modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --phase-order cache_first \
  --output-dir results/modal-vllm-prefix-cache-long-control-paired-cache-first

modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare \
  --prefix-cache-cold-first-paired-dir results/modal-vllm-prefix-cache-long-control-paired \
  --prefix-cache-cache-first-paired-dir results/modal-vllm-prefix-cache-long-control-paired-cache-first \
  --output-dir results/modal-vllm-prefix-cache-long-control-phase-order

modal run modal_app.py --mode vllm-prefix-cache-profile-control \
  --prefix-cache-phase-order-compare-dir results/modal-vllm-prefix-cache-long-control-phase-order \
  --output-dir results/modal-vllm-prefix-cache-long-control-profile-control

# Repeat the paired run with --scenario-seed 569 and trial2 output dirs, then:
modal run modal_app.py --mode vllm-prefix-cache-profile-multitrial \
  --prefix-cache-profile-control-dirs results/modal-vllm-prefix-cache-long-control-profile-control,results/modal-vllm-prefix-cache-long-control-profile-control-trial2 \
  --output-dir results/modal-vllm-prefix-cache-long-control-multitrial
```

Run a small metrics-enabled paired prefix-cache smoke:

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 4 \
  --repeats 1 \
  --warmup-runs 1 \
  --scenario-seed 570 \
  --phase-order cold_first \
  --cache-metrics on \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-metrics-paired-smoke

modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 4 \
  --repeats 1 \
  --warmup-runs 1 \
  --scenario-seed 570 \
  --phase-order cache_first \
  --cache-metrics on \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-metrics-paired-smoke-cache-first

modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare \
  --prefix-cache-cold-first-paired-dir results/modal-vllm-prefix-cache-metrics-paired-smoke \
  --prefix-cache-cache-first-paired-dir results/modal-vllm-prefix-cache-metrics-paired-smoke-cache-first \
  --output-dir results/modal-vllm-prefix-cache-metrics-phase-order-smoke

modal run modal_app.py --mode vllm-prefix-cache-profile-control \
  --prefix-cache-phase-order-compare-dir results/modal-vllm-prefix-cache-metrics-phase-order-smoke \
  --output-dir results/modal-vllm-prefix-cache-metrics-profile-control-smoke
```

Run a small repeated metrics-enabled prefix-cache trial:

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 2 \
  --warmup-runs 1 \
  --scenario-seed 571 \
  --phase-order cold_first \
  --cache-metrics on \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-metrics-repeated

modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 2 \
  --warmup-runs 1 \
  --scenario-seed 571 \
  --phase-order cache_first \
  --cache-metrics on \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-metrics-repeated-cache-first

modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare \
  --prefix-cache-cold-first-paired-dir results/modal-vllm-prefix-cache-metrics-repeated \
  --prefix-cache-cache-first-paired-dir results/modal-vllm-prefix-cache-metrics-repeated-cache-first \
  --output-dir results/modal-vllm-prefix-cache-metrics-repeated-phase-order

modal run modal_app.py --mode vllm-prefix-cache-profile-control \
  --prefix-cache-phase-order-compare-dir results/modal-vllm-prefix-cache-metrics-repeated-phase-order \
  --output-dir results/modal-vllm-prefix-cache-metrics-repeated-profile-control
```

Run a fresh-engine isolated cache-metrics trial:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-metrics \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4 \
  --repeats 1 \
  --scenario-seed 572 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-metrics
```

Run the repeated isolated cache-metrics stability trial:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-metrics \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 2 \
  --scenario-seed 573 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-metrics-repeated
```

Generate the report-ready isolated cache stability summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-metrics-repeated \
  --output-dir results/modal-vllm-prefix-cache-isolated-stability-summary
```

Run a warmed-window isolated cache trial. This adds one throwaway warmup
scenario run inside each fresh cold/cache engine before the measured scenario:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-warm-window \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 1 \
  --scenario-seed 574 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-warm-window
```

Generate a compact summary for that warmed-window trial:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-warm-window \
  --output-dir results/modal-vllm-prefix-cache-isolated-warm-window-summary
```

Run the neutral-warmup isolated cache trial. This warms each fresh engine with
`neutral_long` prompts instead of the measured shared/control prompt bodies:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 1 \
  --scenario-seed 575 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup
```

Generate the neutral-warmup summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup \
  --output-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-summary
```

Generate a measured-window estimate from a neutral-warmup run that includes
warmup-before cache metrics:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-window-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source \
  --output-dir results/modal-vllm-prefix-cache-isolated-window-summary
```

Run the direct-counter neutral-warmup trial. This records vLLM
`CachingMetrics` query/hit deltas around each measured scenario:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 1 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters
```

Generate the counter-aware summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters \
  --output-dir results/modal-vllm-prefix-cache-isolated-counter-summary
```

Run the repeated direct-counter stability trial and summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 3 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability \
  --output-dir results/modal-vllm-prefix-cache-isolated-counter-stability-summary
```

Run a variant prompt-family smoke so repeats can vary the workload family:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long_variant,matched_unique_prefix_variant \
  --output-tokens 8 \
  --request-counts 4 \
  --repeats 2 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-smoke

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-variant-smoke \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-smoke-summary
```

Run the full variant stability grid and summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long_variant,matched_unique_prefix_variant \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 3 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-stability

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-variant-stability \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-stability-summary
```

Audit prompt token and cache-block alignment for the variant grid:

```bash
modal run modal_app.py --mode vllm-prefix-cache-prompt-audit \
  --prompt-profiles shared_prefix_long_variant,matched_unique_prefix_variant \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 3 \
  --scenario-seed 577 \
  --kv-cache-block-size 16 \
  --output-dir results/modal-vllm-prefix-cache-prompt-audit-variant-stability
```

Run the n=16 variant timing probe and summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long_variant,matched_unique_prefix_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 3 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-n16

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-variant-n16 \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-n16-summary
```

Audit prompt token, cache-block, and exact duplicate alignment for the n=16
variant probe:

```bash
modal run modal_app.py --mode vllm-prefix-cache-prompt-audit \
  --prompt-profiles shared_prefix_long_variant,matched_unique_prefix_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 3 \
  --scenario-seed 577 \
  --kv-cache-block-size 16 \
  --output-dir results/modal-vllm-prefix-cache-prompt-audit-variant-n16
```

Run the no-repeat n=16 variant probe, which removes exact duplicate prompts
from the matched control:

```bash
modal run modal_app.py --mode vllm-prefix-cache-prompt-audit \
  --prompt-profiles shared_prefix_long_no_repeat_variant,matched_unique_prefix_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 3 \
  --scenario-seed 577 \
  --kv-cache-block-size 16 \
  --output-dir results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16

modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long_no_repeat_variant,matched_unique_prefix_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 3 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-n16 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-summary
```

Run the no-repeat request-count scaling smoke:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long_no_repeat_variant,matched_unique_prefix_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 8,12,16,20 \
  --repeats 2 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-scaling-smoke

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-scaling-smoke \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-scaling-smoke-summary
```

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

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary
```

Regenerate the no-repeat n=16 stability summary with TTFT/first-event and stream
TPOT profile-control intervals:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft
```

Merge smaller isolated metrics chunks into a single stability source:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-merge \
  --prefix-cache-isolated-merge-dirs results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-smoke-r3,results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r3-seed680,results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r2-seed790 \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8 \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8-summary
```

Modal training is documented in
[`docs/modal-training.md`](docs/modal-training.md).

Run tests:

```bash
python3 -m unittest discover -s tests
```

## Interview Narrative

This repo should let us answer questions like:

- What happens to a request from ingress to final token?
- Which latency metric are we optimizing: TTFT, TPOT, p95, p99, or throughput?
- How do prompt length, output length, and arrival burstiness change queueing?
- What does a benchmark disclose so someone else can reproduce it?
- Where does a microbenchmark result appear, or fail to appear, end to end?

## Current Status

The repo is at Week 1. The simulator is intentionally simple. Its purpose is to
make the measurement model explicit before we attach PyTorch, vLLM, SGLang,
TensorRT-LLM, Triton kernels, or real GPUs.

The research focus is documented in
[`docs/research-focus-kv-cache.md`](docs/research-focus-kv-cache.md).

The first KV-cache pressure baseline is documented in
[`docs/baseline-kv-pressure.md`](docs/baseline-kv-pressure.md).

Workload generation is documented in
[`docs/workload-generation.md`](docs/workload-generation.md).

Scheduler policies are documented in
[`docs/scheduler-policies.md`](docs/scheduler-policies.md).

Capacity-aware scheduling is documented in
[`docs/capacity-aware-scheduling.md`](docs/capacity-aware-scheduling.md).

The first capacity sweep is documented in
[`docs/experiment-001-capacity-sweep.md`](docs/experiment-001-capacity-sweep.md).

The first Modal remote execution path is documented in
[`docs/modal-training.md`](docs/modal-training.md).
