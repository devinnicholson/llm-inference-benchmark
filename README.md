# llm-inference-benchmark-lab

Research-grade learning repo for **568 Systems and Machine Learning**.

The project is now centered on **KV-cache behavior in LLM serving**. The goal is
to build an ML inference systems artifact that can survive an ML infra interview:
clear workload definitions, request lifecycle traces, benchmark methodology,
scheduler experiments, KV-cache pressure studies, and eventually backend
comparisons against real inference engines.

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
