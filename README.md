# KV-Cache Behavior in vLLM Serving

A controlled study of when prefix caching improves LLM serving and when it does not.

This repository contains a benchmark harness for paired cache-on/cache-off experiments across the vLLM OpenAI-compatible server and in-process `AsyncLLM` paths. It records request-level timing, direct prefix-cache counters, GPU KV-cache capacity, and reproducible JSON/CSV/Markdown artifacts.

The current result isolates a simple systems question: under KV-cache pressure, does shared-prefix reuse produce a measurable serving benefit beyond ordinary run-to-run variation?

## Result

The strongest server experiment uses `Qwen/Qwen2.5-1.5B-Instruct` on an NVIDIA L4 with 32 unique requests. The shared-prefix and matched unique-prefix workloads have comparable prompt shapes; the control contains no exact duplicate prompts.

At the lowest successful KV budget (`gpu_memory_utilization=0.325`):

| Workload | Cache hit rate | Throughput, cache on/off | p95 latency, cache on/off |
| --- | ---: | ---: | ---: |
| Shared prefix | 96.204% | 10.741x | 0.095x |
| Matched unique prefix | 0.442% | 0.975x | 1.026x |

Prefix caching reduced p95 latency by 90.5% and increased throughput by 10.7x for the shared-prefix stress workload. The matched control stayed near baseline, which rules out a generic cache-enabled speedup as the explanation.

The result was replicated across two seeds at each successful memory-budget point. The same workload failed vLLM initialization at `gpu_memory_utilization=0.300`, bounding the observed startup floor between `0.300` and `0.325` for this model and serving configuration.

See the [KV-budget report](results/modal-vllm-server-async-qwen15b-l4-kvbudget-report-r1/kv-budget-report.md) for the claim/evidence matrix and the [pressure curve](results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-gpu0325-n32-batched-tokens60640-seed3805-seed3906-pressure-curve-r1/server-cache-pressure-curve.md) for every measured point.

## What the benchmark controls

- Cache enabled versus disabled for the same workload.
- Shared-prefix traffic versus a length-matched unique-prefix control.
- Stable request count, output length, model, GPU class, and scheduler budget.
- Phase order, neutral warmup, prompt identity, and random seed.
- Direct cache-hit counters rather than latency-only inference.
- Exact-duplicate and reusable-block audits for every prompt family.

These controls matter because prefix-cache benchmarks are easy to contaminate with warm caches, duplicate prompts, phase-order effects, or changes in the amount of prefill work.

## System design

```mermaid
flowchart LR
    W["Deterministic workload generator"] --> A["Prompt and KV-block audit"]
    A --> S["vLLM OpenAI-compatible server"]
    A --> E["vLLM AsyncLLM"]
    S --> T["Request traces and server metrics"]
    E --> T
    T --> P["Paired cache and control analysis"]
    P --> R["JSON, CSV, and Markdown artifacts"]
```

The harness includes:

- OpenAI-compatible streaming and concurrent request execution.
- In-process `AsyncLLM` experiments for backend-path comparisons.
- Direct vLLM prefix-cache counter collection.
- TTFT, end-to-end latency, TPOT, throughput, and KV-capacity measurements.
- Phase-order reversal and multitrial aggregation.
- Bootstrap intervals and report generators.
- A deterministic request-lifecycle simulator for scheduler and capacity studies.

## Repository map

| Path | Purpose |
| --- | --- |
| `modal_app.py` | Modal GPU execution, vLLM serving paths, metrics, and experiment orchestration |
| `src/llmbench/` | Workload, scheduling, tracing, and analysis primitives |
| `scripts/` | Workload generation and report builders |
| `configs/` | Model and KV-capacity configurations |
| `workloads/` | Deterministic workload definitions |
| `results/` | Raw measurements and generated analysis artifacts |
| `docs/` | Methodology, experiment history, and limitations |
| `tests/` | Regression tests for workload and report logic |

## Reproduce locally

The local simulator and report builders require Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[modal]"
python -m unittest discover -s tests
```

Generate and replay a deterministic workload:

```bash
python scripts/generate_workload.py mixed_bursty \
  --requests 32 \
  --seed 1337 \
  --output workloads/generated/mixed_bursty_32_seed1337.json

python scripts/replay_workload.py \
  workloads/generated/mixed_bursty_32_seed1337.json \
  --model-config configs/models/llama-7b-gqa-fp16.json \
  --max-concurrent-requests 4 \
  --scheduler-policy memory-aware-deadline
```

GPU experiments run on Modal. Start with a small paired server/`AsyncLLM` check:

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --prompt-profiles short \
  --output-tokens 32 \
  --server-async-prefix-caching off \
  --repeats 3 \
  --warmup-runs 1
```

The complete prefix-cache protocol and canonical artifact paths are documented in [the study report](docs/prefix-cache-study.md).

## Canonical artifacts

- [Primary no-repeat prefix-cache result](results/prefix-cache-study/key-results.md)
- [L4 KV-budget report](results/modal-vllm-server-async-qwen15b-l4-kvbudget-report-r1/kv-budget-report.md)
- [L4 cache-pressure curve](results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu045-gpu040-gpu035-gpu0325-n32-batched-tokens60640-seed3805-seed3906-pressure-curve-r1/server-cache-pressure-curve.md)
- [KV-budget request-count comparison](results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu0325-n16-seed4107-seed4208-vs-n32-request-count-r1/kv-budget-request-count-comparison.md)
- [Full methodology and follow-up studies](docs/prefix-cache-study.md)

## Scope and limitations

The 10.7x throughput result is not a claim about general production traffic. It comes from a deliberately high-overlap, high-pressure workload designed to expose prefix-cache behavior. The current strongest result uses one model family, one GPU class, one request shape, and two seeds per memory-budget point.

The matched unique-prefix control is the important boundary: it remains near baseline while the shared-prefix workload improves. Broader traffic mixtures, additional models, and other inference engines remain separate experiments rather than implied conclusions.
