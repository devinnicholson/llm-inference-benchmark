# Modal Training 001: Remote Sweeps

## Goal

Learn Modal by moving one existing experiment off the laptop without changing
the experiment logic. The first target is CPU-only remote execution of the
capacity sweep. GPU-backed inference comes after this path is boring and
reproducible.

## Concepts

Modal has four pieces we care about first:

- `App`: the namespace that groups functions and deployments.
- `Image`: the container environment and files shipped to remote workers.
- `Function`: code that executes remotely and scales independently.
- `local_entrypoint`: local orchestration code invoked by `modal run`.

This repo's first Modal app is `modal_app.py`.

## What It Runs

The app ships these local directories into the remote image:

- `src/llmbench` -> `/root/llmbench`
- `configs` -> `/root/configs`
- `workloads` -> `/root/workloads`

Then the remote function runs the same `run_sweep` API used by
`scripts/run_sweep.py`.

Default remote sweep:

- Workload: `workloads/generated/mixed_bursty_32_seed568.json`
- Model: `llama-7b-gqa-fp16`
- Capacity: `tight-1gb-kv`
- Concurrency: `4`
- Policies: `deadline`, `memory-aware-deadline`

That intentionally runs only two cases. The goal is to validate the Modal path
with a cheap smoke test before running wider sweeps or GPU workloads.

## Commands

Check the local client and active profile:

```bash
modal --version
modal profile current
```

If the machine is not authenticated:

```bash
modal token new
```

Run the default Modal smoke sweep:

```bash
modal run modal_app.py
```

Write to a different local result directory:

```bash
modal run modal_app.py --output-dir results/modal-training-smoke
```

Run a wider remote sweep:

```bash
modal run modal_app.py \
  --capacities tight-1gb-kv,tight-1-2gb-kv,a10g-24gb-7b-gqa \
  --concurrency 1,2,4,8 \
  --policies fifo,shortest-cache,deadline,memory-aware-deadline \
  --output-dir results/modal-capacity-sweep
```

## Expected Output

The local entrypoint writes:

- `sweep-results.json`
- `sweep-results.csv`

The default run should produce two cases:

- `deadline`
- `memory-aware-deadline`

## Smoke Result

The first smoke run writes to `results/modal-training-smoke`.

| Policy | p95 latency ms | Peak KV MiB | p95 KV MiB | Budget exceeded ms | Deadline miss rate | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deadline` | 3736.294 | 1197.375 | 1150.875 | 625.775 | 0.000 | 1531.548 |
| `memory-aware-deadline` | 3736.294 | 1067.750 | 1021.250 | 197.510 | 0.000 | 1531.548 |

The numbers match the same two rows in the local capacity sweep. That validates
that Modal is currently an execution transport, not a different experimental
implementation.

## Why This Comes Before GPU

For this project, Modal is the execution substrate. The research artifact is
still the KV-cache scheduler and benchmark methodology. Starting with remote
CPU sweeps lets us validate packaging, arguments, returned artifacts, and
reproducibility before adding GPU dependencies such as PyTorch, vLLM, SGLang,
TensorRT-LLM, or Triton kernels.

# Modal Training 002: GPU Probe

## Goal

Prove that this repo can allocate a real Modal GPU, inspect the CUDA stack, run
a tiny PyTorch CUDA kernel, and save the probe result as a reproducible artifact.

This is still not model inference. It is a deliberately small bridge between
remote CPU sweeps and GPU-backed serving experiments.

## Command

```bash
modal run modal_app.py --mode gpu-probe
```

The probe currently requests a `T4` GPU. That is enough to validate CUDA access
while keeping the first GPU run small.

## Output

The local entrypoint writes:

- `results/modal-gpu-probe/probe.json`

The probe records:

- Modal execution marker
- Python version
- PyTorch version
- PyTorch CUDA runtime version
- `torch.cuda.is_available()`
- `nvidia-smi` query output
- GPU device name
- GPU memory
- compute capability
- one `1024x1024` FP16 matmul wall-clock timing

## Probe Result

The first successful run wrote `results/modal-gpu-probe/probe.json`.

| Field | Value |
| --- | --- |
| CUDA available | `true` |
| Device | `Tesla T4` |
| GPU memory | `14912.6875 MiB` |
| Compute capability | `7.5` |
| NVIDIA driver | `580.95.05` |
| CUDA runtime | `13.0` |
| PyTorch | `2.12.0+cu130` |
| Matmul smoke | `1024x1024_fp16` |
| Matmul wall time | `111.511 ms` |

The first attempted run also exposed a Modal serialization rule that matters for
future experiments: remote functions should return JSON-native values or local
code must have matching deserialization dependencies. Returning
`str(torch.__version__)` instead of the Torch version object keeps the local
environment free of Torch.

## Why This Matters

The synthetic scheduler is useful only if we can later compare it with real
backend behavior. A working GPU probe proves the repo can now run on the same
kind of remote accelerator that a real vLLM, SGLang, TensorRT-LLM, or Triton
experiment would need.

## Next Step After Probe

Add a tiny real inference run on Modal GPU. The first backend should be chosen
for packaging simplicity, not final performance. A small Hugging Face
`transformers` run is acceptable if it gets us clean TTFT and decode timing
hooks quickly. After that, move to vLLM or SGLang.

# Modal Training 003: Tiny Inference Timing

## Goal

Run the first real GPU-backed text generation path and record timing fields that
map back to the simulator vocabulary: prompt tokens, generated tokens, TTFT,
decode time, TPOT, output tokens/sec, and peak allocated GPU memory.

This is not a serving benchmark yet. It is a single-request backend smoke test
using Hugging Face Transformers.

## Model

Default model:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
```

This was chosen because it is small enough for cheap Modal training runs while
still using a real chat template, tokenizer, causal LM, KV cache, and CUDA
execution path.

## Command

```bash
modal run modal_app.py --mode tiny-inference
```

Override the prompt or generation length:

```bash
modal run modal_app.py \
  --mode tiny-inference \
  --prompt "Explain why KV cache memory grows with sequence length." \
  --max-new-tokens 64
```

## Modal Pieces Learned

This milestone adds:

- a separate inference image with `torch`, `transformers`, `accelerate`,
  `safetensors`, and `numpy`
- a Modal `Volume` mounted at `/cache`
- `HF_HOME=/cache` so Hugging Face model files persist between runs
- a `tiny-inference` local entrypoint mode
- JSON-only return values for local deserialization safety

The first run downloads model files into the volume. Later runs can reuse the
cache, which is why the recorded final artifact has lower load times than the
uncached first attempt.

## Measurement Method

The remote function avoids the high-level `pipeline()` helper so the timing
points are explicit:

1. Load tokenizer.
2. Load model in `float16` on `cuda:0`.
3. Tokenize the prompt using the model chat template.
4. Run a short warmup decode.
5. Run one manual greedy prefill/first-token step with `use_cache=True`.
6. Decode the remaining tokens one at a time using `past_key_values`.
7. Record TTFT, decode time, TPOT, total generation time, throughput, and peak
   allocated GPU memory.

`ttft_ms` here is a single-request proxy for prefill plus first-token selection.
`tpot_ms` is computed over tokens after the first generated token.

## Result

The first committed result is
`results/modal-tiny-inference/inference.json`.

| Field | Value |
| --- | --- |
| Model | `HuggingFaceTB/SmolLM2-135M-Instruct` |
| GPU | `Tesla T4` |
| Prompt tokens | `43` |
| Generated tokens | `32` |
| TTFT | `37.221 ms` |
| Decode time | `1005.135 ms` |
| TPOT | `32.424 ms` |
| Output tokens/sec | `30.700` |
| Peak allocated GPU memory | `270.084 MiB` |
| Tokenizer load | `381.488 ms` |
| Model load | `1512.843 ms` |
| Warmup | `907.659 ms` |

Generated text:

```text
A KV cache in LLM is a data structure that stores the results of a LLM computation, allowing for efficient retrieval of the results of subsequent computations.
```

## Interpretation

This gives us the first bridge from synthetic scheduling to a real backend. The
numbers are not comparable to vLLM or SGLang yet because this path is a
single-request manual decode loop in Transformers. That limitation is useful:
it makes the next step clear.

The next training milestone should run the same prompt through a backend with a
serving-oriented scheduler and paged/block KV cache, then compare measurement
fields against this raw Transformers baseline.

# Modal Training 004: vLLM Baseline

## Goal

Run the same prompt through vLLM, a serving-oriented inference engine, and
compare the result against the raw Transformers baseline.

This is still not an online serving benchmark. It uses vLLM's offline
`LLM.generate` path first because that is the smallest reliable bridge from a
single-request Transformers loop to a backend with paged KV cache, chunked
prefill, prefix caching, and scheduler configuration.

## Command

```bash
modal run modal_app.py --mode vllm-inference
```

The first committed vLLM result is:

```text
results/modal-vllm-inference/vllm-inference.json
```

## Modal Pieces Learned

This milestone adds:

- a CUDA 12.9 vLLM image based on `nvidia/cuda:12.9.0-devel-ubuntu22.04`
- `vllm==0.21.0`
- a separate Modal Volume for vLLM cache artifacts
- `VLLM_CACHE_ROOT=/vllm-cache`
- a `vllm-inference` local entrypoint mode
- a constrained smoke-test vLLM config:
  - `max_model_len=1024`
  - `max_num_batched_tokens=1024`
  - `max_num_seqs=1`
  - `gpu_memory_utilization=0.50`
  - `enforce_eager=True`

## Result

| Field | Transformers manual decode | vLLM offline generate |
| --- | ---: | ---: |
| Backend | `transformers` | `vllm 0.21.0` |
| GPU | `Tesla T4` | `Tesla T4` |
| Prompt tokens | `43` | `43` |
| Generated tokens | `32` | `30` |
| Generation wall time | `1042.356 ms` | `1563.241 ms` |
| Output tokens/sec | `30.700` | `19.191` |
| TTFT | `37.221 ms` | not exposed by offline artifact |
| TPOT | `32.424 ms` | not exposed by offline artifact |

vLLM generated:

```text
A KV cache in LLM is a data structure that stores the results of a LLM computation, allowing for efficient data retrieval and manipulation.
```

## vLLM Runtime Observations

The logs are more important than the single-request throughput number:

- vLLM resolved the model architecture as `LlamaForCausalLM`.
- The model weights were `bfloat16` and were cast to `float16` on T4.
- FlashAttention 2 was unavailable because T4 compute capability is `7.5`;
  vLLM selected a FlashInfer attention backend instead.
- vLLM reported approximately `6.96 GiB` available for KV cache.
- vLLM reported `324,320` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `316.72x` for `1024` tokens/request.
- Engine initialization, including profile, KV-cache creation, and warmup, took
  `134.47 s`.
- A Triton kernel JIT compilation happened during inference for
  `_compute_slot_mapping_kernel`, causing a latency spike.

## Interpretation

For this tiny single-request run, vLLM is slower than the raw Transformers
manual decode path. That is not surprising. We are paying for a serving engine
whose advantages show up under batching, concurrency, prefix reuse, and memory
pressure, not in one short prompt.

This result is still the right next step for the research artifact because it
gives us a real serving-engine surface:

- explicit scheduler settings
- KV-cache capacity reporting
- backend/hardware compatibility behavior
- warmup and JIT costs
- a baseline that can be compared against concurrent workloads later

## Next Step

Move from offline `LLM.generate` to a small OpenAI-compatible vLLM server mode
with streaming enabled. That should let us measure TTFT directly from the first
streamed token and compare raw Transformers, vLLM offline, and vLLM server
timing on the same prompt.

# Modal Training 005: vLLM Streaming

## Goal

Measure vLLM's async streaming path so we can distinguish first-output latency
from full generation wall time.

This uses vLLM's V1 `AsyncLLM` engine and `RequestOutputKind.DELTA`, which
streams newly generated text chunks as they arrive. It is still in-process
offline inference, not an OpenAI-compatible HTTP server, but it measures the
same first-yield behavior that server streaming depends on.

## Command

```bash
modal run modal_app.py --mode vllm-streaming
```

The first committed streaming result is:

```text
results/modal-vllm-streaming/vllm-streaming.json
```

## Result

| Field | Value |
| --- | ---: |
| Backend | `vllm 0.21.0` |
| GPU | `Tesla T4` |
| Prompt tokens | `43` |
| Generated tokens | `30` |
| Engine load/init | `151070.897 ms` |
| First streamed chunk | `1466.684 ms` |
| Stream wall time | `2072.620 ms` |
| Decode after first chunk | `605.937 ms` |
| Stream TPOT after first chunk | `20.894 ms` |
| Output tokens/sec | `14.474` |

Generated text:

```text
A KV cache in LLM is a data structure that stores the results of a LLM computation, allowing for efficient data retrieval and manipulation.
```

## Comparison

| Field | Transformers manual decode | vLLM offline generate | vLLM async streaming |
| --- | ---: | ---: | ---: |
| Prompt tokens | `43` | `43` | `43` |
| Generated tokens | `32` | `30` | `30` |
| First output | `37.221 ms` | not exposed | `1466.684 ms` |
| TPOT | `32.424 ms` | not exposed | `20.894 ms` |
| Generation wall time | `1042.356 ms` | `1563.241 ms` | `2072.620 ms` |
| Output tokens/sec | `30.700` | `19.191` | `14.474` |

The vLLM streaming result shows lower per-token decode time after the first
chunk than the raw Transformers loop, but a much slower first output. The run
also logged a Triton JIT compilation during inference, which likely inflated the
first streamed chunk time.

## Measurement Caveat

`first_chunk_ms` is the first yielded streaming chunk, not necessarily exactly
one token. In this run, the first chunk contained two token IDs and text
`"A K"`. For API-facing TTFT, this is still the user-visible first output. For a
strict first-token benchmark, the next harness should force or validate
single-token stream intervals.

## Runtime Observations

The streaming run repeated the same serving-engine observations as the offline
vLLM baseline:

- FlashAttention 2 was unavailable on T4 compute capability `7.5`.
- vLLM selected FlashInfer attention.
- vLLM reported approximately `6.96 GiB` available KV-cache memory.
- vLLM reported `324,320` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `316.72x` for `1024` tokens/request.
- Engine initialization took `144.97 s`.
- A Triton JIT compilation happened during inference for
  `_compute_slot_mapping_kernel`.

## Next Step

Now the project has three single-request baselines:

- raw Transformers manual decode
- vLLM offline generate
- vLLM async streaming

The next research step should stop optimizing the single prompt and instead run
a small concurrent workload through vLLM. That is where a serving engine should
start to differ from the raw Transformers baseline in a way that matters for
KV-cache scheduling.

# Modal Training 006: vLLM Concurrent Streaming

## Goal

Run a small simultaneous workload through one vLLM `AsyncLLM` engine and measure
per-request first output, per-request completion latency, and aggregate output
throughput.

This is the first Modal training run that exercises vLLM like a serving engine
rather than a single-prompt inference script.

## Command

```bash
modal run modal_app.py --mode vllm-concurrent
```

Change request count:

```bash
modal run modal_app.py --mode vllm-concurrent --prompt-count 8
```

The first committed concurrent result is:

```text
results/modal-vllm-concurrent/vllm-concurrent.json
```

## Workload

The default concurrent workload uses four short prompts:

- KV-cache pressure in LLM serving
- batching and GPU utilization
- prefill versus decode
- long prompts and tail latency

All four requests are started against the same loaded vLLM engine with:

- `max_model_len=1024`
- `max_num_batched_tokens=2048`
- `max_num_seqs=4`
- `gpu_memory_utilization=0.50`
- `enforce_eager=True`

## Result

| Field | Value |
| --- | ---: |
| Backend | `vllm 0.21.0` |
| GPU | `Tesla T4` |
| Requests | `4` |
| Total output tokens | `128` |
| Batch wall time | `1866.175 ms` |
| Aggregate output tokens/sec | `68.589` |
| p50 first chunk | `678.972 ms` |
| p95 first chunk | `679.163 ms` |
| p50 latency | `1295.784 ms` |
| p95 latency | `1296.581 ms` |
| p50 stream TPOT | `19.894 ms` |
| p95 stream TPOT | `20.526 ms` |
| Engine load/init | `137300.286 ms` |

## Comparison

| Field | vLLM single streaming | vLLM 4-request concurrent |
| --- | ---: | ---: |
| Requests | `1` | `4` |
| Output tokens | `30` | `128` |
| Wall time | `2072.620 ms` | `1866.175 ms` |
| Output tokens/sec | `14.474` | `68.589` |
| First output p95 | `1466.684 ms` | `679.163 ms` |
| Completion p95 | `2072.620 ms` | `1296.581 ms` |
| TPOT p95 | `20.894 ms` | `20.526 ms` |

The prompt set is not identical to the single-prompt streaming run, so this is
not a controlled speedup claim. It is still the first strong signal that vLLM's
serving engine becomes more interesting under concurrency: aggregate throughput
improves materially while per-token decode time remains in the same range.

## Runtime Observations

The concurrent run repeated the same hardware/backend behavior:

- FlashAttention 2 was unavailable on T4 compute capability `7.5`.
- vLLM selected FlashInfer attention.
- vLLM reported approximately `6.95 GiB` available KV-cache memory.
- vLLM reported `323,824` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `316.23x` for `1024` tokens/request.
- Engine initialization took `129.62 s`.
- A Triton JIT compilation happened during inference for
  `_compute_slot_mapping_kernel`.

## Interpretation

This is the first result that connects back to the simulator's original purpose.
The synthetic scheduler studies were about how request concurrency, deadlines,
and KV-cache pressure interact. The concurrent vLLM run shows where real serving
engines expose the same surfaces:

- prompt token counts
- generated token counts
- first output latency
- tail completion latency
- aggregate throughput
- KV-cache capacity reported by the engine
- scheduler limits such as `max_num_seqs` and `max_num_batched_tokens`

## Next Step

Turn this into a real experiment sweep:

- run prompt counts `1`, `2`, `4`, and `8`
- keep the prompt set and generation length fixed
- compare p95 first output, p95 completion latency, aggregate throughput, and
  vLLM-reported KV-cache capacity
- save one CSV/JSON result table instead of one ad hoc JSON artifact

That sweep will finally give a clean bridge between the synthetic capacity
sweep and real vLLM serving behavior.

# Modal Training 007: vLLM Concurrency and Context Sweep

## Goal

Turn the one-off concurrent vLLM run into a controlled experiment table. This
sweep varies:

- request count: `1`, `2`, `4`, `8`
- prompt profile: `short`, `long`
- output budget: `16`, `32`

The result is the first real backend artifact that can be plotted against the
synthetic scheduler work: throughput, first output, tail completion latency,
TPOT, prompt tokens, output tokens, and estimated live sequence tokens.

## Command

```bash
modal run modal_app.py --mode vllm-sweep
```

The committed result writes:

```text
results/modal-vllm-sweep/vllm-sweep.json
results/modal-vllm-sweep/vllm-sweep.csv
```

This first version was a single-run sweep. The current artifact format is
superseded by Modal Training 008 below, which adds repeats, seeded shuffle, and
per-repeat CSV output.

Override the grid:

```bash
modal run modal_app.py \
  --mode vllm-sweep \
  --request-counts 1,4,8 \
  --prompt-profiles short,long,mixed \
  --output-tokens 32
```

## Method

One Modal `T4` worker loads one vLLM `AsyncLLM` engine with:

- `vllm==0.21.0`
- `max_model_len=1024`
- `max_num_batched_tokens=8192`
- `max_num_seqs=8`
- `gpu_memory_utilization=0.50`
- `enable_prefix_caching=False`
- `enforce_eager=True`

Prefix caching is disabled for this sweep so repeated prompts across request
counts do not turn into a prefix-cache benchmark by accident. The run performs a
one-token shape warmup for each scenario before measurement, then records the
measured scenario grid.

The JSON artifact keeps per-request timings and generated text. The CSV artifact
keeps one flat row per scenario for plotting and spreadsheet inspection.

## Runtime Observations

The corrected sweep logged:

- FlashAttention 2 unavailable on T4 compute capability `7.5`.
- FlashInfer selected as the attention backend.
- vLLM reported `6.88 GiB` available KV-cache memory.
- vLLM reported `320,400` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `312.89x` for `1024` tokens/request.
- Engine initialization took `152.06 s`.
- Shape warmup generated `60` tokens across `16` warmup scenarios in
  `1683.817 ms`.

## Result: 32 Output Tokens

The `32` token output budget is the cleanest first comparison because decode
time dominates more than the very short `16` token cases.

| Profile | Requests | Prompt tokens mean | Peak sequence tokens | p95 first chunk ms | p95 latency ms | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 44.00 | 76 | 42.206 | 658.706 | 48.564 |
| short | 2 | 43.50 | 151 | 66.472 | 695.793 | 91.951 |
| short | 4 | 43.25 | 301 | 68.193 | 719.836 | 177.745 |
| short | 8 | 45.25 | 618 | 70.246 | 730.611 | 350.206 |
| long | 1 | 118.00 | 150 | 43.115 | 660.063 | 48.462 |
| long | 2 | 117.50 | 299 | 66.416 | 691.602 | 92.508 |
| long | 4 | 117.25 | 597 | 67.200 | 693.981 | 184.368 |
| long | 8 | 119.25 | 1210 | 69.801 | 715.642 | 357.475 |

## Result: 16 Output Tokens

The `16` token output budget is noisier because fixed overhead and remaining
shape effects are a larger share of the request. The outliers are kept in the
artifact instead of hidden.

| Profile | Requests | Prompt tokens mean | Peak sequence tokens | p95 first chunk ms | p95 latency ms | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 44.00 | 60 | 642.418 | 965.841 | 16.561 |
| short | 2 | 43.50 | 119 | 87.118 | 468.528 | 68.252 |
| short | 4 | 43.25 | 237 | 101.082 | 444.810 | 143.713 |
| short | 8 | 45.25 | 490 | 95.863 | 408.578 | 312.949 |
| long | 1 | 118.00 | 134 | 43.648 | 333.320 | 47.972 |
| long | 2 | 117.50 | 267 | 65.617 | 358.687 | 89.152 |
| long | 4 | 117.25 | 533 | 282.954 | 588.966 | 108.610 |
| long | 8 | 119.25 | 1082 | 69.469 | 368.329 | 347.151 |

## Interpretation

For `32` output tokens, aggregate throughput scales nearly linearly from one to
eight concurrent requests on this tiny model while p95 completion latency only
increases modestly. The short profile moves from `48.564` to `350.206` output
tokens/sec, and the long profile moves from `48.462` to `357.475` output
tokens/sec.

The long prompts are roughly 2.7x the token length of the short prompts, but at
this model size and context length they do not dominate the `32` token results.
Decode batching is the main effect we can see.

The `16` token cases show why single-run microbenchmarks are dangerous. Two
scenarios have visible first-output outliers even after shape warmup:

- `short_out16_n1`: `642.418 ms` p95 first chunk
- `long_out16_n4`: `282.954 ms` p95 first chunk

Those outliers are useful, not embarrassing. They tell us the next benchmark
needs repetitions, randomized scenario order, and confidence intervals before
we claim stable performance curves.

## Next Step

Training 008 adds repeat support and turns this from a first benchmark table
into a more defensible experiment.

# Modal Training 008: Repeated vLLM Sweep

## Goal

Make the concurrency/context sweep less dependent on a single run order. The
same 16 scenarios now run with:

- `3` measured repeats per scenario
- seeded random scenario order with seed `568`
- one aggregate CSV row per scenario
- one run-level CSV row per measured scenario execution
- median, p95, min, max, mean, and coefficient of variation across repeats

This is still not a statistically complete benchmark, but it is a meaningful
upgrade from one measurement per scenario.

## Command

```bash
modal run modal_app.py --mode vllm-sweep --repeats 3 --scenario-seed 568
```

The current repeated artifact writes:

```text
results/modal-vllm-sweep/vllm-sweep.json
results/modal-vllm-sweep/vllm-sweep.csv
results/modal-vllm-sweep/vllm-sweep-runs.csv
```

## Method Changes

The engine settings are still:

- `max_model_len=1024`
- `max_num_batched_tokens=8192`
- `max_num_seqs=8`
- `gpu_memory_utilization=0.50`
- `enable_prefix_caching=False`
- `enforce_eager=True`

The run still performs one-token shape warmups before measurement. After warmup,
it expands the scenario grid into `48` measured runs and shuffles that run plan
with seed `568`.

## Runtime Observations

The repeated sweep logged:

- FlashInfer selected as the attention backend on T4.
- vLLM reported `6.88 GiB` available KV-cache memory.
- vLLM reported `320,400` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `312.89x` for `1024` tokens/request.
- Engine initialization took `156.93 s`.
- Shape warmup generated `60` tokens across `16` scenarios in `1765.127 ms`.

## Repeated Result: 32 Output Tokens

Median values across three repeats:

| Profile | Requests | Prompt tokens mean | Peak sequence tokens | Median output tok/s | Median p95 first chunk ms | Median p95 latency ms | Throughput CV | Latency CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 44.00 | 76 | 44.118 | 43.641 | 725.110 | 0.043 | 0.042 |
| short | 2 | 43.50 | 151 | 82.016 | 80.396 | 779.962 | 0.090 | 0.092 |
| short | 4 | 43.25 | 301 | 172.321 | 75.088 | 742.276 | 0.093 | 0.099 |
| short | 8 | 45.25 | 618 | 328.727 | 86.379 | 778.291 | 0.052 | 0.053 |
| long | 1 | 118.00 | 150 | 42.429 | 57.854 | 753.957 | 0.040 | 0.039 |
| long | 2 | 117.50 | 299 | 79.287 | 83.727 | 806.955 | 0.041 | 0.040 |
| long | 4 | 117.25 | 597 | 156.908 | 86.116 | 815.424 | 0.148 | 0.158 |
| long | 8 | 119.25 | 1210 | 273.144 | 118.478 | 936.751 | 0.072 | 0.076 |

## Repeated Result: 16 Output Tokens

Median values across three repeats:

| Profile | Requests | Prompt tokens mean | Peak sequence tokens | Median output tok/s | Median p95 first chunk ms | Median p95 latency ms | Throughput CV | Latency CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short | 1 | 44.00 | 60 | 43.182 | 46.923 | 368.175 | 0.102 | 0.099 |
| short | 2 | 43.50 | 119 | 72.647 | 82.545 | 440.167 | 0.372 | 0.505 |
| short | 4 | 43.25 | 237 | 118.524 | 135.904 | 538.787 | 0.103 | 0.097 |
| short | 8 | 45.25 | 490 | 290.304 | 88.002 | 440.269 | 0.027 | 0.026 |
| long | 1 | 118.00 | 134 | 42.336 | 48.847 | 377.738 | 0.013 | 0.013 |
| long | 2 | 117.50 | 267 | 79.005 | 66.483 | 404.736 | 0.065 | 0.062 |
| long | 4 | 117.25 | 533 | 144.310 | 94.868 | 443.163 | 0.034 | 0.034 |
| long | 8 | 119.25 | 1082 | 299.199 | 77.761 | 426.862 | 0.049 | 0.046 |

## Interpretation

The repeated `32` token results still show the main serving-engine effect:
throughput rises with concurrency while latency grows much more slowly than
throughput. For short prompts, median throughput moves from `44.118` to
`328.727` output tokens/sec from one to eight concurrent requests. For long
prompts, it moves from `42.429` to `273.144`.

The repeat data also weakens the overly clean single-run story. Long prompts at
`8` concurrent requests now have higher median p95 latency than the short
profile: `936.751 ms` versus `778.291 ms`. That is the shape we expected to see
as live sequence tokens grow.

The coefficient of variation fields are now the most useful benchmark-quality
signal. `short_out16_n2` is unstable with throughput CV `0.372` and latency CV
`0.505`, so that scenario should not be used for conclusions without more
repeats. The `32` token cases are generally more stable, though `long_out32_n4`
still has visible variance.

## Next Step

Training 009 adds a paired prefix-cache sweep.

# Modal Training 009: Prefix-Cache Comparison

## Goal

Run the same repeated sweep with vLLM prefix caching enabled and compare it
against the cold-prefix control from Training 008.

This experiment intentionally uses repeated prompts. In Training 008, repeated
prompts were controlled by disabling prefix caching. In this training run, the
same repetition becomes the feature under study.

## Commands

Run the prefix-cache sweep:

```bash
modal run modal_app.py \
  --mode vllm-sweep \
  --prefix-caching on \
  --repeats 3 \
  --scenario-seed 568
```

Generate the paired comparison:

```bash
modal run modal_app.py --mode vllm-prefix-cache-compare
```

The prefix-cache sweep writes:

```text
results/modal-vllm-prefix-cache-sweep/vllm-sweep.json
results/modal-vllm-prefix-cache-sweep/vllm-sweep.csv
results/modal-vllm-prefix-cache-sweep/vllm-sweep-runs.csv
```

The comparison writes:

```text
results/modal-vllm-prefix-cache-compare/prefix-cache-compare.json
results/modal-vllm-prefix-cache-compare/prefix-cache-compare.csv
```

## Method

The prefix-cache run uses the same grid, repeats, seed, model, GPU, and scheduler
limits as the cold-prefix control:

- `3` measured repeats per scenario
- seed `568`
- request counts `1`, `2`, `4`, `8`
- prompt profiles `short`, `long`
- output budgets `16`, `32`
- `max_model_len=1024`
- `max_num_batched_tokens=8192`
- `max_num_seqs=8`
- `enable_prefix_caching=True`

The comparison joins the cold and cached aggregate CSVs by `scenario_id` and
reports median throughput, median p95 first chunk, median p95 latency, median
p95 TPOT, and repeat-level coefficient of variation.

## Runtime Observations

The prefix-cache sweep logged:

- vLLM engine config had `enable_prefix_caching=True`.
- FlashInfer selected as the attention backend on T4.
- vLLM reported `6.88 GiB` available KV-cache memory.
- vLLM reported `320,400` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `312.89x` for `1024` tokens/request.
- Engine initialization took `147.18 s`.
- Shape warmup generated `60` tokens across `16` scenarios in `1375.001 ms`.

## Result: 32 Output Tokens

Ratios compare prefix-cache enabled against the cold-prefix control. A throughput
ratio above `1.0` is better. First chunk, latency, and TPOT ratios below `1.0`
are better.

| Profile | Requests | Peak sequence tokens | Throughput ratio | First chunk ratio | Latency ratio | TPOT ratio | Prefix throughput CV | Prefix latency CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long | 1 | 150 | 1.184 | 0.727 | 0.845 | 0.864 | 0.005 | 0.005 |
| long | 2 | 299 | 1.202 | 0.759 | 0.832 | 0.845 | 0.008 | 0.008 |
| long | 4 | 597 | 1.191 | 0.763 | 0.840 | 0.846 | 0.143 | 0.159 |
| long | 8 | 1210 | 1.362 | 0.582 | 0.734 | 0.728 | 0.020 | 0.020 |
| short | 1 | 76 | 1.156 | 0.951 | 0.865 | 0.858 | 0.007 | 0.007 |
| short | 2 | 151 | 1.162 | 0.784 | 0.860 | 0.865 | 0.008 | 0.008 |
| short | 4 | 301 | 1.109 | 0.869 | 0.902 | 0.907 | 0.005 | 0.005 |
| short | 8 | 618 | 1.142 | 0.777 | 0.876 | 0.890 | 0.002 | 0.002 |

Across the eight `32` token scenarios, prefix caching averaged:

- `1.188x` throughput ratio
- `0.844x` p95 latency ratio
- `0.776x` p95 first chunk ratio

## Result: 16 Output Tokens

| Profile | Requests | Peak sequence tokens | Throughput ratio | First chunk ratio | Latency ratio | TPOT ratio | Prefix throughput CV | Prefix latency CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| long | 1 | 134 | 1.183 | 0.857 | 0.845 | 0.853 | 0.007 | 0.007 |
| long | 2 | 267 | 1.163 | 0.973 | 0.860 | 0.856 | 0.008 | 0.008 |
| long | 4 | 533 | 1.259 | 0.691 | 0.793 | 0.815 | 0.006 | 0.006 |
| long | 8 | 1082 | 1.206 | 0.872 | 0.830 | 0.857 | 0.012 | 0.012 |
| short | 1 | 60 | 1.145 | 0.910 | 0.878 | 0.874 | 0.006 | 0.006 |
| short | 2 | 119 | 1.273 | 0.776 | 0.786 | 0.777 | 0.350 | 0.466 |
| short | 4 | 237 | 1.557 | 0.497 | 0.643 | 0.713 | 0.009 | 0.009 |
| short | 8 | 490 | 1.246 | 0.757 | 0.803 | 0.822 | 0.013 | 0.013 |

Across all `16` scenarios, prefix caching averaged:

- `1.221x` throughput ratio
- `0.825x` p95 latency ratio
- `0.784x` p95 first chunk ratio

## Interpretation

Prefix caching improved every scenario in this paired run on median throughput
and median p95 latency. The largest `32` token improvement was
`long_out32_n8`, where throughput increased by `1.362x`, p95 first chunk fell
to `0.582x`, and p95 latency fell to `0.734x` of the cold-prefix control.

The result is strongest on repeated long prompts, which is exactly where prefix
caching should help: prefill work is reused and the live decode workload becomes
more dominant. Short prompts still improve, but the ceiling is smaller because
there is less prefill work to avoid.

The caution is that this benchmark is cache-warmed by design. It should not be
reported as a general online serving speedup unless the workload actually has
prefix reuse. It is a prefix-reuse benchmark, not a random-prompt benchmark.

## Next Step

Training 010 moves to an OpenAI-compatible vLLM server smoke.

# Modal Training 010: OpenAI-Compatible vLLM Server Streaming

## Goal

Move from in-process `AsyncLLM` calls to the vLLM OpenAI-compatible HTTP server.
This is the first benchmark artifact that exercises a production-shaped API
surface: server startup, `/health`, `/v1/chat/completions`, SSE streaming, and
OpenAI-style usage accounting.

This is still a smoke test, not a server benchmark. It sends one streaming chat
request after the server is healthy.

## Command

```bash
modal run modal_app.py --mode vllm-server-streaming
```

The committed result writes:

```text
results/modal-vllm-server-streaming/vllm-server-streaming.json
```

## Method

The Modal worker starts:

```bash
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 1024 \
  --max-num-batched-tokens 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.50 \
  --enforce-eager
```

The local worker logic then:

1. Starts the server as a subprocess.
2. Captures server logs into the JSON artifact.
3. Polls `GET /health` until the server is ready.
4. Sends `POST /v1/chat/completions` with `stream=true`.
5. Parses SSE `data:` events and records first content, total stream wall time,
   TPOT, chunks, generated text, and OpenAI-compatible usage.
6. Terminates the server process.

## Runtime Observations

The server run logged:

- vLLM server started on `http://127.0.0.1:8000`.
- `/health` returned `200 OK`.
- `/v1/chat/completions` returned `200 OK`.
- vLLM exposed `/metrics`, `/v1/models`, `/v1/chat/completions`,
  `/v1/completions`, `/v1/responses`, `/tokenize`, and `/detokenize`.
- FlashInfer selected as the attention backend on T4.
- vLLM reported `6.96 GiB` available KV-cache memory.
- vLLM reported `324,320` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `316.72x` for `1024` tokens/request.
- Engine initialization took `125.32 s`.
- Server health was ready after `161326.565 ms`.
- A Triton JIT compilation happened during the streamed request for
  `_compute_slot_mapping_kernel`.

## Result

| Field | Value |
| --- | ---: |
| Backend | `vllm-openai-server 0.21.0` |
| GPU | `Tesla T4` |
| Prompt tokens | `43` |
| Completion tokens | `30` |
| Total tokens | `73` |
| Server ready | `161326.565 ms` |
| First content | `925.424 ms` |
| Request wall time | `1500.791 ms` |
| Decode after first content | `575.367 ms` |
| Stream TPOT | `19.840 ms` |
| Output tokens/sec | `19.989` |
| SSE chunks | `31` |

Generated text:

```text
A KV cache in LLM is a data structure that stores the results of a LLM computation, allowing for efficient data retrieval and manipulation.
```

## Interpretation

The server path gives us two metrics that the earlier in-process runs did not:

- API readiness latency: model load, profiling, KV-cache allocation, server
  startup, route registration, and health availability.
- API-facing streaming latency: the user-visible delay between sending a chat
  completions request and receiving the first content-bearing SSE event.

The first content time was `925.424 ms`, lower than the earlier single-request
`AsyncLLM` streaming first chunk result, but this is not a controlled speedup
claim. The server smoke uses a different process boundary and captures one run.
Its value is that the measurement now matches the API surface an ML infra system
would expose.

The server artifact also confirms that OpenAI-compatible usage accounting is
available for this route: `43` prompt tokens, `30` completion tokens, and `73`
total tokens.

## Next Step

Training 011 adds a server-side concurrent streaming workload against
`/v1/chat/completions`.

# Modal Training 011: OpenAI-Compatible vLLM Server Concurrent Streaming

## Goal

Move the server path from a single smoke request to a small concurrency sweep.
This milestone starts one vLLM OpenAI-compatible server, waits for `/health`,
then sends `1`, `2`, `4`, and `8` concurrent streaming chat completions through
`/v1/chat/completions`.

The point is to measure the API-facing version of the same serving questions we
have been building toward: first-content latency, p95 request latency, TPOT,
aggregate output throughput, usage accounting, and the server's own KV-cache
capacity logs.

## Command

```bash
modal run modal_app.py --mode vllm-server-concurrent
```

The default prompt profile is `short`. Use `--prompt-profile long` or
`--prompt-profile mixed` to run the same server path against different prompt
sets.

The committed result writes:

```text
results/modal-vllm-server-concurrent/vllm-server-concurrent.json
results/modal-vllm-server-concurrent/vllm-server-concurrent.csv
```

## Method

The Modal worker starts:

```bash
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 1024 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.50 \
  --enforce-eager
```

The local worker logic then:

- start one vLLM server
- wait for `/health`
- send `1`, `2`, `4`, and `8` concurrent streaming chat requests
- record per-request first content, p95 latency, TPOT, token usage, and aggregate
  throughput
- capture server logs around KV-cache capacity, route registration, and request
  handling

Each request uses `stream=true` and `stream_options.include_usage=true`, so the
artifact records both client-observed streaming timing and OpenAI-compatible
usage accounting.

## Runtime Observations

The server run logged:

- vLLM server started on `http://127.0.0.1:8000`.
- `/health` returned `200 OK`.
- `/v1/chat/completions` returned `200 OK` for the streamed requests.
- vLLM reported `6.88 GiB` available KV-cache memory.
- vLLM reported `320,400` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `312.89x` for `1024` tokens/request.
- Engine initialization took `132.00 s`.
- Server health was ready after `165325.553 ms`.
- A Triton JIT compilation happened during inference for
  `_compute_slot_mapping_kernel`.

## Result

| Requests | Peak sequence tokens | Batch wall ms | p95 first content ms | p95 latency ms | p95 TPOT ms | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 76 | 1590.024 | 1026.232 | 1589.735 | 18.178 | 20.125 |
| 2 | 151 | 673.024 | 76.884 | 672.887 | 19.870 | 95.093 |
| 4 | 301 | 663.540 | 99.438 | 663.109 | 18.807 | 192.905 |
| 8 | 618 | 837.285 | 256.552 | 836.748 | 19.282 | 305.750 |

## Interpretation

The API-server path shows the concurrency effect directly: aggregate throughput
increases from about `20` output tokens/sec at one request to about `306` output
tokens/sec at eight concurrent requests.

The single-request first-content number is high because this is the first
measured server request after readiness and includes post-startup JIT work. The
later concurrent scenarios have much lower first-content latency. That makes
scenario order and warmup policy an important methodology issue for any claim
we make from server-side benchmarks.

At eight concurrent requests, p95 request latency stayed under `837 ms` while
p95 streaming TPOT stayed near `19 ms`. Compared with the earlier in-process
`AsyncLLM` repeated prefix-cache sweep, server throughput is lower than the
`short_out32_n8` median of `375.332` output tokens/sec, but this is not a
controlled server-vs-in-process comparison. The server path includes HTTP, SSE,
JSON parsing, one run per scenario, and scenario-order effects.

## Next Step

Training 012 adds the repeated server sweep with warmup/discard runs and seeded
scenario ordering.

# Modal Training 012: Repeated vLLM Server Sweep

## Goal

Turn the one-shot server concurrent run into a repeatable benchmark shape. This
milestone keeps one OpenAI-compatible vLLM server alive, runs warmup requests
that are discarded, then measures a seeded and shuffled set of server-side
concurrency scenarios.

The key methodology change is that first-request JIT and route-level startup
effects are no longer mixed directly into the first measured scenario.

## Command

```bash
modal run modal_app.py --mode vllm-server-sweep \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568
```

The committed result writes:

```text
results/modal-vllm-server-sweep/vllm-server-sweep.json
results/modal-vllm-server-sweep/vllm-server-sweep.csv
results/modal-vllm-server-sweep/vllm-server-sweep-runs.csv
```

## Method

This run uses the same server configuration as Training 011:

```bash
vllm serve HuggingFaceTB/SmolLM2-135M-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 1024 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.50 \
  --enforce-eager
```

The measured grid is intentionally focused:

- Prompt profile: `short`
- Output tokens: `32`
- Concurrent request counts: `1`, `2`, `4`, `8`
- Warmup passes: `1`
- Measured repeats: `3`
- Scenario order seed: `568`

The warmup pass sends one-token streaming requests for every shape and discards
those timings. The measured pass then shuffles the scenario plan and records
both aggregate rows and per-run rows.

## Runtime Observations

The server run logged:

- vLLM server started on `http://127.0.0.1:8000`.
- `/health` returned `200 OK`.
- `/v1/chat/completions` returned `200 OK` for all warmup and measured
  requests.
- vLLM reported `6.88 GiB` available KV-cache memory.
- vLLM reported `320,400` GPU KV-cache tokens.
- vLLM reported maximum concurrency of `312.89x` for `1024` tokens/request.
- Engine initialization took `136.33 s`.
- Server health was ready after `168349.012 ms`.
- The warmup pass took `789.441 ms` across `4` scenario shapes and `15`
  generated warmup tokens.
- Triton JIT for `_compute_slot_mapping_kernel` appeared during the warmup
  requests rather than inside the first measured one-request scenario.

## Result

| Requests | Repeats | Median batch wall ms | Repeat p95 batch wall ms | Median p95 first-content ms | Repeat p95 first-content ms | Median p95 latency ms | Repeat p95 latency ms | Median p95 TPOT ms | Median output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 664.399 | 691.134 | 51.586 | 51.745 | 663.513 | 690.301 | 19.789 | 48.164 |
| 2 | 3 | 688.926 | 1163.510 | 74.549 | 75.688 | 687.936 | 1162.658 | 20.454 | 92.898 |
| 4 | 3 | 697.589 | 717.109 | 78.515 | 92.006 | 695.757 | 716.070 | 20.178 | 183.489 |
| 8 | 3 | 733.485 | 871.041 | 97.006 | 226.732 | 732.474 | 869.423 | 21.296 | 349.019 |

## Interpretation

Warmup materially changes the server story. The one-shot Training 011
single-request first-content value was `1026.232 ms`; in this repeated run the
single-request median p95 first-content value is `51.586 ms`. That confirms the
earlier one-shot result was polluted by first measured request effects.

Throughput scales from `48.164` median output tokens/sec at one request to
`349.019` at eight concurrent requests. That is close to the earlier in-process
`AsyncLLM` short `32` token, eight-request median of `375.332` output tokens/sec,
but the server comparison is still not controlled enough to claim parity: the
server path includes HTTP, SSE, JSON parsing, and a different client loop.

The `n=2` case had one measured latency outlier: `rep02-run000` reported
`1162.658 ms` p95 latency and `35.751 ms` p95 TPOT while the other `n=2` runs
were near `659-688 ms`. Keeping the per-run CSV is useful because it makes this
visible instead of hiding it inside an average.

## Next Step

Training 013 adds a local comparison artifact between the repeated server sweep
and the matched in-process `AsyncLLM` sweep.

# Modal Training 013: Server vs AsyncLLM Comparison

## Goal

Compare the repeated OpenAI-compatible server sweep against the existing
in-process `AsyncLLM` sweep for the overlapping scenarios. This is an artifact
comparison, not a final overhead benchmark: it compares two separate Modal runs
with matching scenario definitions.

The purpose is to make the next fair experiment obvious. If separate runs do
not show the expected API overhead, we need a paired A/B run before making any
claim about server transport cost.

## Command

```bash
modal run modal_app.py --mode vllm-server-sweep-compare
```

The committed result writes:

```text
results/modal-vllm-server-sweep-compare/server-vs-async.json
results/modal-vllm-server-sweep-compare/server-vs-async.csv
```

## Compared Artifacts

The comparison reads:

```text
results/modal-vllm-sweep/vllm-sweep.csv
results/modal-vllm-server-sweep/vllm-server-sweep.csv
```

The overlapping scenario set is:

- Prompt profile: `short`
- Output tokens: `32`
- Concurrent request counts: `1`, `2`, `4`, `8`
- Repeats per scenario: `3`

## Result

| Requests | Async tok/s | Server tok/s | Server/Async tok/s | Async first event ms | Server first content ms | Server/Async first event | Async p95 latency ms | Server p95 latency ms | Server/Async latency | Async p95 TPOT ms | Server p95 TPOT ms | Server/Async TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 44.118 | 48.164 | 1.092 | 43.641 | 51.586 | 1.182 | 725.110 | 663.513 | 0.915 | 22.061 | 19.789 | 0.897 |
| 2 | 82.016 | 92.898 | 1.133 | 80.396 | 74.549 | 0.927 | 779.962 | 687.936 | 0.882 | 23.317 | 20.454 | 0.877 |
| 4 | 172.321 | 183.489 | 1.065 | 75.088 | 78.515 | 1.046 | 742.276 | 695.757 | 0.937 | 22.179 | 20.178 | 0.910 |
| 8 | 328.727 | 349.019 | 1.062 | 86.379 | 97.006 | 1.123 | 778.291 | 732.474 | 0.941 | 22.961 | 21.296 | 0.927 |

Mean ratios across the four matched scenarios:

- Server/Async throughput: `1.088`
- Server/Async first event: `1.070`
- Server/Async p95 latency: `0.919`
- Server/Async p95 TPOT: `0.903`

## Interpretation

This comparison does not show a simple HTTP/SSE overhead penalty. The server
artifact is slightly higher throughput on all four matched scenarios and lower
p95 latency on all four matched scenarios, while first-event latency is worse on
three of four scenarios.

That does not mean the server is intrinsically faster than in-process
`AsyncLLM`. These rows come from separate Modal runs, so they include run-to-run
GPU variance, scenario order differences, different client loops, server warmup
behavior, and different measurement boundaries. The result is best interpreted
as a methodology finding: artifact-level CSV comparison is useful for spotting
large differences, but not sufficient to isolate API transport overhead.

## Next Step

Training 014 adds the paired same-worker benchmark with matched prompts,
warmups, scenario order, and per-run deltas.

# Modal Training 014: Paired Server vs AsyncLLM Benchmark

## Goal

Run the in-process `AsyncLLM` path and the OpenAI-compatible server path inside
one Modal worker, then compare matching `(scenario_id, repeat_index)` pairs.
This removes a major weakness from Training 013: the server and async rows are
no longer from unrelated Modal jobs.

This is still not the final overhead benchmark because the phase order is fixed:
`AsyncLLM` runs first, then the server starts second on the same worker.

## Command

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568
```

The committed result writes:

```text
results/modal-vllm-server-async-paired/paired-server-async.json
results/modal-vllm-server-async-paired/paired-server-async-summary.csv
results/modal-vllm-server-async-paired/paired-server-async-runs.csv
```

## Method

The paired runner:

- Builds one scenario plan from the seed `568`.
- Runs the in-process `AsyncLLM` phase first.
- Runs one one-token warmup pass and discards those timings.
- Measures `3` repeats for each `short_out32_n{1,2,4,8}` scenario.
- Shuts down the async engine, runs garbage collection, and clears CUDA cache.
- Starts `vllm serve` on the same worker.
- Runs the same warmup and the same measured scenario plan.
- Pairs rows by `(scenario_id, repeat_index)`.

## Runtime Observations

The paired run logged:

- Async engine load: `154248.278 ms`.
- Server ready after Async shutdown: `33099.757 ms`.
- Async warmup: `966.939 ms` across `4` shapes and `15` generated tokens.
- Server warmup: `362.018 ms` across `4` shapes and `15` generated tokens.
- Server-side vLLM engine init was only `2.80 s`.
- Server-side KV-cache capacity matched previous runs: `320,400` GPU KV-cache
  tokens and `312.89x` maximum concurrency for `1024` tokens/request.

The very short server init is useful, but it is also a caveat: the server phase
benefits from running second in an already-warmed worker.

## Result

The ratio columns are medians of paired per-run ratios. Throughput ratios above
`1.0` favor the server. Latency, first-event, and TPOT ratios below `1.0` favor
the server.

| Requests | Pairs | Server/Async tok/s | Server/Async first event | Server/Async p95 latency | Server/Async TPOT | Async median tok/s | Server median tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0.995 | 1.081 | 1.004 | 0.998 | 47.597 | 48.275 |
| 2 | 3 | 1.074 | 1.221 | 0.931 | 0.936 | 88.713 | 86.063 |
| 4 | 3 | 1.082 | 1.170 | 0.924 | 0.900 | 173.843 | 187.393 |
| 8 | 3 | 1.098 | 1.332 | 0.910 | 0.879 | 333.102 | 359.460 |

Mean ratios across all `12` paired runs:

- Server/Async throughput: `1.099`
- Server/Async first event: `1.299`
- Server/Async p95 latency: `0.943`
- Server/Async p95 TPOT: `0.909`

## Interpretation

The paired run strengthens the finding from Training 013: this setup does not
show a large server throughput penalty. Server throughput ratios are near or
above `1.0`, and p95 latency plus TPOT ratios are usually below `1.0`.

The server's first-event latency is consistently worse, especially at higher
concurrency. That is the metric most likely to expose API/SSE/client-loop
overhead in this benchmark shape.

There are still paired outliers. In `short_out32_n2_rep02`, AsyncLLM was much
slower than the server, producing a server/async throughput ratio of `1.903` and
a latency ratio of `0.524`. In `short_out32_n8_rep02`, the server had a
first-event outlier with a ratio of `2.716` and a latency ratio of `1.149`.

The main caveat is phase order. Because AsyncLLM runs first, then the server
runs after model files, kernels, and process-local state have already been
warmed, this paired benchmark is not enough to claim that the server path is
faster. It tells us the next control we need.

## Next Step

Training 015 adds the `server_first` phase-order control and compares it against
the `async_first` paired result.

# Modal Training 015: Phase-Order Control

## Goal

Test whether the paired Training 014 result depends on phase order. Training
014 ran `AsyncLLM` first and the server second. This milestone runs the same
paired benchmark with `server_first`, using the same prompts, warmup policy,
repeats, and scenario seed.

## Command

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order server_first \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568
```

The committed result writes:

```text
results/modal-vllm-server-async-paired-server-first/paired-server-async.json
results/modal-vllm-server-async-paired-server-first/paired-server-async-summary.csv
results/modal-vllm-server-async-paired-server-first/paired-server-async-runs.csv
```

## Runtime Observations

The server-first run logged:

- Server ready: `173620.882 ms`.
- Server-side vLLM engine init: `141.37 s`.
- Async engine load after server shutdown: `21686.931 ms`.
- Server warmup: `896.157 ms` across `4` shapes and `15` generated tokens.
- Async warmup: `272.285 ms` across `4` shapes and `15` generated tokens.
- Server-side KV-cache capacity again matched previous runs: `320,400` GPU
  KV-cache tokens and `312.89x` maximum concurrency for `1024` tokens/request.

This reverses the warm-state relationship from Training 014. In Training 014,
the server ran second and reached readiness in `33099.757 ms`. Here, the server
runs first and pays the cold startup cost.

## Server-First Result

The ratio columns are medians of paired per-run ratios.

| Requests | Pairs | Server/Async tok/s | Server/Async first event | Server/Async p95 latency | Server/Async TPOT | Async median tok/s | Server median tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3 | 0.903 | 1.405 | 1.107 | 1.069 | 49.024 | 44.109 |
| 2 | 3 | 0.843 | 1.273 | 1.185 | 1.144 | 85.538 | 80.133 |
| 4 | 3 | 0.902 | 1.482 | 1.108 | 1.077 | 186.599 | 168.521 |
| 8 | 3 | 0.924 | 1.368 | 1.081 | 1.058 | 352.692 | 325.904 |

Mean ratios across all `12` paired runs:

- Server/Async throughput: `0.880`
- Server/Async first event: `1.409`
- Server/Async p95 latency: `1.154`
- Server/Async p95 TPOT: `1.133`

## Phase-Order Comparison

| Phase order | Server ready ms | Async engine load ms | Mean Server/Async tok/s | Mean Server/Async first event | Mean Server/Async p95 latency | Mean Server/Async TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | 33099.757 | 154248.278 | 1.099 | 1.299 | 0.943 | 0.909 |
| `server_first` | 173620.882 | 21686.931 | 0.880 | 1.409 | 1.154 | 1.133 |

## Interpretation

The phase-order control flips the conclusion. When the server runs second, it
looks slightly better on throughput, p95 latency, and TPOT. When the server runs
first, it is worse on all four mean ratio metrics, including throughput.

That means the Training 014 server advantage was not a stable backend claim. It
was at least partly a warm-state and phase-order effect. The first-event result
is the most stable signal: the server is worse in both orders, and the
server-first control makes that penalty larger.

This is a strong methodology improvement for the artifact. We now have evidence
that same-worker paired benchmarks must be counterbalanced before they are used
to argue about API transport overhead.

## Next Step

Training 016 adds the compact machine-readable phase-order comparison artifact.

# Modal Training 016: Phase-Order Comparison Artifact

## Goal

Make the Training 014 and Training 015 phase-order result reproducible without
manually reading two large JSON files. This milestone reads the `async_first`
and `server_first` paired outputs and writes one comparison table.

## Command

```bash
modal run modal_app.py --mode vllm-server-async-phase-order-compare
```

The committed result writes:

```text
results/modal-vllm-server-async-phase-order-compare/phase-order-compare.json
results/modal-vllm-server-async-phase-order-compare/phase-order-compare.csv
```

## Result

| Requests | Async-first tok/s ratio | Server-first tok/s ratio | Delta | Async-first latency ratio | Server-first latency ratio | Delta | Async-first first-event ratio | Server-first first-event ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.995 | 0.903 | -0.092 | 1.004 | 1.107 | 0.103 | 1.081 | 1.405 |
| 2 | 1.074 | 0.843 | -0.231 | 0.931 | 1.185 | 0.254 | 1.221 | 1.273 |
| 4 | 1.082 | 0.902 | -0.181 | 0.924 | 1.108 | 0.184 | 1.170 | 1.482 |
| 8 | 1.098 | 0.924 | -0.174 | 0.910 | 1.081 | 0.171 | 1.332 | 1.368 |

Mean ratios:

| Phase order | Server ready ms | Async engine load ms | Server/Async tok/s | Server/Async first event | Server/Async p95 latency | Server/Async TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `async_first` | 33099.757 | 154248.278 | 1.099 | 1.299 | 0.943 | 0.909 |
| `server_first` | 173620.882 | 21686.931 | 0.880 | 1.409 | 1.154 | 1.133 |

## Interpretation

The compact artifact makes the order effect unambiguous:

- Every scenario has a lower server/async throughput ratio when the server runs
  first.
- Every scenario has a higher server/async p95 latency ratio when the server
  runs first.
- First-event latency is worse for the server in both phase orders.

This turns the paired benchmark into a useful research artifact: it does not
just report a speed number, it identifies a confounder and records the evidence
in machine-readable form.

## Next Step

Training 017 adds a second independent phase-order trial across fresh Modal
workers.

# Modal Training 017: Second Phase-Order Trial

## Goal

Start estimating whether the phase-order effect from Training 016 is stable
across fresh Modal workers. This milestone repeats both paired phase orders in
new output directories and generates a second phase-order comparison artifact.

## Commands

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order async_first \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --output-dir results/modal-vllm-server-async-paired-async-first-trial2
```

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order server_first \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --output-dir results/modal-vllm-server-async-paired-server-first-trial2
```

```bash
modal run modal_app.py --mode vllm-server-async-phase-order-compare \
  --async-first-paired-dir results/modal-vllm-server-async-paired-async-first-trial2 \
  --server-first-paired-dir results/modal-vllm-server-async-paired-server-first-trial2 \
  --output-dir results/modal-vllm-server-async-phase-order-compare-trial2
```

## Artifacts

```text
results/modal-vllm-server-async-paired-async-first-trial2/paired-server-async.json
results/modal-vllm-server-async-paired-server-first-trial2/paired-server-async.json
results/modal-vllm-server-async-phase-order-compare-trial2/phase-order-compare.json
results/modal-vllm-server-async-phase-order-compare-trial2/phase-order-compare.csv
```

## Result

Mean server/async ratios:

| Trial | Async-first tok/s | Server-first tok/s | Delta | Async-first latency | Server-first latency | Delta | Async-first TPOT | Server-first TPOT | Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.099 | 0.880 | -0.219 | 0.943 | 1.154 | 0.211 | 0.909 | 1.133 | 0.224 |
| 2 | 1.047 | 0.981 | -0.066 | 0.975 | 1.040 | 0.065 | 0.938 | 1.007 | 0.069 |

First-event ratios:

| Trial | Async-first first event | Server-first first event | Delta |
| ---: | ---: | ---: | ---: |
| 1 | 1.299 | 1.409 | 0.110 |
| 2 | 1.365 | 1.357 | -0.008 |

## Interpretation

Trial 2 repeats the direction of the throughput, latency, and TPOT order effect,
but the magnitude is smaller than Trial 1. That means phase order is real, but
the size of the effect is noisy at this sample count.

The first-event result remains the most robust server penalty: both phase
orders in both trials have server/async first-event ratios above `1.0`.

The current evidence is now better than a single benchmark number:

- `async_first` makes the server look better on throughput and p95 latency.
- `server_first` makes the server look worse on throughput and p95 latency.
- The effect is visible across two independent trial pairs, but the magnitude
  changes enough that we should not report a single overhead number yet.

## Next Step

Training 018 adds the multi-trial phase-order aggregate.

# Modal Training 018: Multi-Trial Phase-Order Aggregate

## Goal

Summarize the first two phase-order comparison artifacts in one machine-readable
table. This is the first artifact that treats the phase-order effect as a
distribution across independent trial pairs instead of a single result.

## Command

```bash
modal run modal_app.py --mode vllm-server-async-multitrial-aggregate
```

The first aggregate input was:

```text
results/modal-vllm-server-async-phase-order-compare
results/modal-vllm-server-async-phase-order-compare-trial2
```

The committed result writes:

```text
results/modal-vllm-server-async-multitrial-aggregate/phase-order-multitrial.json
results/modal-vllm-server-async-multitrial-aggregate/phase-order-multitrial.csv
```

## Initial Result

| Metric | Async-first mean | Async-first min | Async-first max | Server-first mean | Server-first min | Server-first max | Server-first minus async-first mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Throughput ratio | 1.073 | 1.047 | 1.099 | 0.931 | 0.880 | 0.981 | -0.142 |
| First-event ratio | 1.332 | 1.299 | 1.365 | 1.383 | 1.357 | 1.409 | 0.051 |
| Latency ratio | 0.959 | 0.943 | 0.975 | 1.097 | 1.040 | 1.154 | 0.138 |
| TPOT ratio | 0.923 | 0.909 | 0.938 | 1.070 | 1.007 | 1.133 | 0.146 |

## Interpretation

Across the first two independent trial pairs:

- `async_first` averages above `1.0` for throughput ratio and below `1.0` for
  latency and TPOT ratios.
- `server_first` averages below `1.0` for throughput ratio and above `1.0` for
  latency and TPOT ratios.
- First-event ratio is above `1.0` in both phase orders, which remains the most
  consistent server penalty.

The order-effect direction is stable across the two trials for throughput,
latency, and TPOT. The magnitude is not stable enough to report one final
server-overhead number. The signed delta CVs are large because there are only
two trial pairs.

## Next Step

Training 019 adds one more trial pair and regenerates this aggregate with
bootstrap intervals for the order effect.

# Modal Training 019: Third Trial and Bootstrap Aggregate

## Goal

Run a third independent phase-order trial pair, then regenerate the multi-trial
aggregate with bootstrap percentile intervals for the signed order effect. This
makes the benchmark less dependent on one lucky or unlucky fresh Modal worker.

## Commands

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order async_first \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --output-dir results/modal-vllm-server-async-paired-async-first-trial3
```

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order server_first \
  --prompt-profiles short \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --output-dir results/modal-vllm-server-async-paired-server-first-trial3
```

```bash
modal run modal_app.py --mode vllm-server-async-phase-order-compare \
  --async-first-paired-dir results/modal-vllm-server-async-paired-async-first-trial3 \
  --server-first-paired-dir results/modal-vllm-server-async-paired-server-first-trial3 \
  --output-dir results/modal-vllm-server-async-phase-order-compare-trial3
```

```bash
modal run modal_app.py --mode vllm-server-async-multitrial-aggregate
```

The aggregate default now reads:

```text
results/modal-vllm-server-async-phase-order-compare
results/modal-vllm-server-async-phase-order-compare-trial2
results/modal-vllm-server-async-phase-order-compare-trial3
```

## Artifacts

```text
results/modal-vllm-server-async-paired-async-first-trial3/paired-server-async.json
results/modal-vllm-server-async-paired-server-first-trial3/paired-server-async.json
results/modal-vllm-server-async-phase-order-compare-trial3/phase-order-compare.json
results/modal-vllm-server-async-multitrial-aggregate/phase-order-multitrial.json
results/modal-vllm-server-async-multitrial-aggregate/phase-order-multitrial.csv
```

## Trial 3 Result

Mean server/async ratios:

| Phase order | Server/Async tok/s | Server/Async first event | Server/Async p95 latency | Server/Async TPOT |
| --- | ---: | ---: | ---: | ---: |
| `async_first` | 1.178 | 1.095 | 0.903 | 0.892 |
| `server_first` | 0.982 | 1.275 | 1.044 | 1.018 |

## Three-Trial Aggregate

The interval columns are deterministic bootstrap percentile intervals over the
mean signed order effect using `4096` resamples. They are not final confidence
claims; with only three trial pairs they are a compact uncertainty check.

| Metric | Async-first mean | Server-first mean | Delta mean | Bootstrap p05 | Bootstrap p50 | Bootstrap p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Throughput ratio | 1.108 | 0.948 | -0.160 | -0.211 | -0.160 | -0.109 |
| First-event ratio | 1.253 | 1.347 | 0.094 | 0.031 | 0.094 | 0.157 |
| Latency ratio | 0.940 | 1.079 | 0.139 | 0.090 | 0.139 | 0.188 |
| TPOT ratio | 0.913 | 1.053 | 0.140 | 0.088 | 0.140 | 0.191 |

## Interpretation

The third trial reinforces the phase-order effect:

- The server/async throughput ratio is higher in `async_first` than
  `server_first` in all three trial pairs.
- The server/async latency and TPOT ratios are lower in `async_first` than
  `server_first` in all three trial pairs.
- First-event latency remains above `1.0` for the server in both phase orders.

The bootstrap intervals still come from a small sample, but they no longer cross
zero for throughput, latency, or TPOT order-effect means. That is enough to say
phase order is a real confounder in this benchmark harness. It is not enough to
publish one universal server-overhead number.

## Next Step

Training 020 repeats the counterbalanced benchmark for `long` prompts, where
prefill and prefix behavior should matter more than the current short-prompt
decode-heavy workload.

# Modal Training 020: Long-Prompt Phase-Order Benchmark

## Goal

Test whether the short-prompt phase-order finding holds when prompt lengths are
larger. This run uses the same paired benchmark harness, seed, warmup policy,
request counts, and output length, but switches `--prompt-profiles` from
`short` to `long`.

## Commands

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order async_first \
  --prompt-profiles long \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --output-dir results/modal-vllm-server-async-paired-long-async-first
```

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --phase-order server_first \
  --prompt-profiles long \
  --output-tokens 32 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --output-dir results/modal-vllm-server-async-paired-long-server-first
```

```bash
modal run modal_app.py --mode vllm-server-async-phase-order-compare \
  --async-first-paired-dir results/modal-vllm-server-async-paired-long-async-first \
  --server-first-paired-dir results/modal-vllm-server-async-paired-long-server-first \
  --output-dir results/modal-vllm-server-async-phase-order-compare-long
```

## Artifacts

```text
results/modal-vllm-server-async-paired-long-async-first/paired-server-async.json
results/modal-vllm-server-async-paired-long-server-first/paired-server-async.json
results/modal-vllm-server-async-phase-order-compare-long/phase-order-compare.json
results/modal-vllm-server-async-phase-order-compare-long/phase-order-compare.csv
```

## Runtime Observations

The long async-first run reported:

- Async engine load: `176828.958 ms`
- Server ready after async phase: `65351.094 ms`
- Mean server/async throughput ratio: `1.024`
- Mean server/async p95 latency ratio: `1.001`

The long server-first run reported:

- Server ready: `168297.585 ms`
- Async engine load after server phase: `20254.661 ms`
- Mean server/async throughput ratio: `0.904`
- Mean server/async p95 latency ratio: `1.125`

Both runs used `long_out32_n{1,2,4,8}` scenarios with `3` paired repeats.

## Result

The ratio columns are medians of paired per-run ratios for each scenario.

| Requests | Async-first tok/s ratio | Server-first tok/s ratio | Async-first first-event ratio | Server-first first-event ratio | Async-first p95 latency ratio | Server-first p95 latency ratio | Async-first TPOT ratio | Server-first TPOT ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.005 | 0.969 | 1.469 | 1.226 | 0.986 | 1.031 | 0.944 | 1.020 |
| 2 | 0.922 | 0.979 | 1.375 | 1.154 | 1.081 | 1.021 | 1.036 | 1.007 |
| 4 | 0.970 | 0.900 | 1.244 | 1.324 | 1.030 | 1.110 | 1.110 | 1.070 |
| 8 | 0.982 | 0.899 | 1.365 | 1.448 | 1.017 | 1.112 | 0.965 | 1.073 |

Mean ratios:

| Phase order | Server/Async tok/s | Server/Async first event | Server/Async p95 latency | Server/Async TPOT |
| --- | ---: | ---: | ---: | ---: |
| `async_first` | 1.024 | 1.333 | 1.001 | 0.978 |
| `server_first` | 0.904 | 1.433 | 1.125 | 1.091 |

## Interpretation

Long prompts preserve the phase-order story, but they change the magnitude.
When the server runs second, the throughput ratio is only slightly above `1.0`
and p95 latency is essentially neutral. When the server runs first, throughput
and latency are clearly worse than `AsyncLLM`.

First-event latency remains the most stable server penalty. It is above `1.0`
for both phase orders and every long-prompt request count.

Compared with the short-prompt aggregate, the long-prompt workload makes the
server's async-first advantage much weaker. That is useful evidence that prompt
profile matters and that the artifact should not publish a single overhead
number without workload stratification.

## Next Step

Training 021 adds a short-vs-long workload comparison artifact that reads the
short-prompt three-trial aggregate and the long-prompt comparison artifact,
then reports how the phase-order effect changes with prompt profile.

# Modal Training 021: Short-vs-Long Workload Comparison

## Goal

Turn the prompt-profile observation from Training 020 into a compact artifact.
This is a local comparison step: it does not launch another GPU benchmark. It
reads the short-prompt three-trial phase-order aggregate and the long-prompt
phase-order comparison, then reports how the mean server/async ratios change.

## Command

```bash
modal run modal_app.py --mode vllm-server-async-workload-compare
```

## Artifacts

```text
results/modal-vllm-server-async-workload-compare/workload-compare.json
results/modal-vllm-server-async-workload-compare/workload-compare.csv
```

## Sources

```text
results/modal-vllm-server-async-multitrial-aggregate/phase-order-multitrial.json
results/modal-vllm-server-async-phase-order-compare-long/phase-order-compare.json
```

## Result

`Delta` is `server_first_mean - async_first_mean`. Positive latency-like deltas
mean the server-first phase order made the server path worse relative to
`AsyncLLM`; negative throughput deltas mean the server-first phase order reduced
server throughput relative to `AsyncLLM`.

| Metric | Short async-first | Long async-first | Long-short async | Short server-first | Long server-first | Long-short server | Short delta | Long delta | Long-short delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Throughput ratio | 1.108 | 1.024 | -0.084 | 0.948 | 0.904 | -0.044 | -0.160 | -0.121 | 0.039 |
| First-event ratio | 1.253 | 1.333 | 0.080 | 1.347 | 1.433 | 0.086 | 0.094 | 0.100 | 0.007 |
| p95 latency ratio | 0.940 | 1.001 | 0.060 | 1.079 | 1.125 | 0.046 | 0.139 | 0.125 | -0.015 |
| TPOT ratio | 0.913 | 0.978 | 0.065 | 1.053 | 1.091 | 0.039 | 0.140 | 0.113 | -0.026 |

## Interpretation

Long prompts weaken the async-first server advantage. In the short aggregate,
async-first showed server throughput above `AsyncLLM` and p95 latency below
`AsyncLLM`; in the long run, throughput is only `1.024x` and p95 latency is
effectively neutral at `1.001x`.

Server-first remains the stricter test. Throughput is below `AsyncLLM` for both
short and long prompts, and p95 latency stays above `AsyncLLM` for both prompt
profiles.

First-event latency is the most consistent server penalty. The long-prompt run
increases the first-event ratio in both phase orders, and the short-vs-long
delta is nearly identical for async-first and server-first.

The main research implication is that prompt profile is now a required
stratification axis. The artifact should report server overhead by workload
shape and phase order instead of collapsing the data into one universal
server/async number.

## Next Step

Training 022 shifts from interface overhead toward a KV-cache-specific pilot:
shared-prefix workloads with prefix caching on and off.

# Modal Training 022: Shared-Prefix KV-Cache Pilot

## Goal

Create an explicit prefix-reuse workload instead of relying on accidental prompt
repetition. The new `shared_prefix` profile gives every request the same long
incident brief and changes only the final task suffix. This should make the
cacheability assumption visible in the artifact.

This is still a two-run comparison, not a same-worker paired control. It is a
pilot for the next, stricter experiment.

## Commands

```bash
modal run modal_app.py --mode vllm-sweep \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --scenario-seed 568 \
  --prefix-caching off \
  --output-dir results/modal-vllm-shared-prefix-cold
```

```bash
modal run modal_app.py --mode vllm-sweep \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --scenario-seed 568 \
  --prefix-caching on \
  --output-dir results/modal-vllm-shared-prefix-cache
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-compare \
  --cold-sweep-dir results/modal-vllm-shared-prefix-cold \
  --prefix-sweep-dir results/modal-vllm-shared-prefix-cache \
  --output-dir results/modal-vllm-shared-prefix-cache-compare
```

## Artifacts

```text
results/modal-vllm-shared-prefix-cold/vllm-sweep.json
results/modal-vllm-shared-prefix-cold/vllm-sweep.csv
results/modal-vllm-shared-prefix-cache/vllm-sweep.json
results/modal-vllm-shared-prefix-cache/vllm-sweep.csv
results/modal-vllm-shared-prefix-cache-compare/prefix-cache-compare.json
results/modal-vllm-shared-prefix-cache-compare/prefix-cache-compare.csv
```

## Runtime Observations

Both runs used the same `shared_prefix_out32_n{1,2,4,8}` scenarios with `3`
repeats and seed `568`.

- Cold-prefix engine load: `145944.214 ms`
- Prefix-cache engine load: `150721.970 ms`
- Cold warmup wall time: `781.469 ms`
- Prefix-cache warmup wall time: `852.615 ms`
- Prompt tokens per request: about `226`
- `max_model_len=1024`
- `max_num_batched_tokens=8192`
- `max_num_seqs=8`

## Result

Ratios compare prefix caching on against prefix caching off. Throughput ratios
above `1.0` are better. First-token, latency, and TPOT ratios below `1.0` are
better.

| Requests | Peak sequence tokens | Throughput ratio | First-token ratio | p95 latency ratio | TPOT ratio | Cold throughput CV | Prefix throughput CV |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 258 | 0.958 | 1.069 | 1.044 | 1.040 | 0.017 | 0.050 |
| 2 | 517 | 0.993 | 0.993 | 1.007 | 1.016 | 0.255 | 0.275 |
| 4 | 1034 | 0.960 | 1.023 | 1.041 | 1.042 | 0.018 | 0.008 |
| 8 | 2043 | 0.990 | 1.040 | 1.011 | 1.014 | 0.029 | 0.039 |

Mean ratios across the four scenarios:

- Throughput ratio: `0.975`
- First-token ratio: `1.032`
- p95 latency ratio: `1.026`
- TPOT ratio: `1.028`

## Interpretation

This pilot is a negative result. The explicit shared-prefix workload did not
show a prefix-cache win. Median throughput was slightly lower with prefix
caching enabled, and latency-like metrics were slightly worse.

That does not invalidate the broader KV-cache direction. It means this
two-run comparison is not strong enough to support a speedup claim. Training
009 showed large gains on repeated ordinary prompts, while this run shows no
gain on explicit shared-prefix prompts. The difference could be workload shape,
cache warmup behavior, vLLM prefix-cache overhead on a small T4 model, or
ordinary cross-run variance.

The useful research outcome is methodological: KV-cache experiments need the
same phase-order discipline we added for server-vs-`AsyncLLM`. A separate cold
Modal run and a separate cached Modal run are not enough.

## Next Step

Training 023 adds a same-worker paired prefix-cache benchmark. It runs cold and
cached `AsyncLLM` engines in both phase orders, writes paired rows by
`scenario_id` and `repeat_index`, then compares `cold_first` against
`cache_first`.

# Modal Training 023: Paired Prefix-Cache Phase-Order Control

## Goal

Replace the separate-run Training 022 pilot with a same-worker paired benchmark.
Each Modal job loads two `AsyncLLM` engines in one worker, runs the same seeded
scenario plan through both engines, and pairs rows by `scenario_id` and
`repeat_index`.

This controls the biggest weakness in the prior prefix-cache artifacts:
comparing one cold Modal run against a separate cached Modal run.

## Commands

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --phase-order cold_first
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --phase-order cache_first
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare
```

## Artifacts

```text
results/modal-vllm-prefix-cache-paired/paired-prefix-cache.json
results/modal-vllm-prefix-cache-paired/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-paired/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-paired-cache-first/paired-prefix-cache.json
results/modal-vllm-prefix-cache-paired-cache-first/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-paired-cache-first/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-phase-order-compare/prefix-cache-phase-order-compare.json
results/modal-vllm-prefix-cache-phase-order-compare/prefix-cache-phase-order-compare.csv
```

## Runtime Observations

The first engine in each Modal worker paid the full vLLM initialization cost.
The second engine initialized much faster after CUDA and process state were
already warm:

| Phase order | Cold engine load | Cache engine load | Paired runs |
| --- | ---: | ---: | ---: |
| `cold_first` | `224443.487 ms` | `45792.795 ms` | 12 |
| `cache_first` | `20086.412 ms` | `152320.607 ms` | 12 |

Both runs used `shared_prefix_out32_n{1,2,4,8}`, `3` paired repeats,
`warmup_runs=1`, and seed `568`.

## Result

Ratios compare prefix caching on against prefix caching off. Throughput ratios
above `1.0` are better. First-event, latency, and TPOT ratios below `1.0` are
better.

Mean paired-run ratios:

| Phase order | Throughput ratio | First-event ratio | p95 latency ratio | TPOT ratio |
| --- | ---: | ---: | ---: | ---: |
| `cold_first` | 1.131 | 1.083 | 0.923 | 0.900 |
| `cache_first` | 1.133 | 0.865 | 0.910 | 0.916 |

Scenario median ratios:

| Requests | Cold-first throughput | Cache-first throughput | Cold-first p95 latency | Cache-first p95 latency | Cold-first TPOT | Cache-first TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.997 | 1.218 | 1.007 | 0.821 | 1.013 | 0.830 |
| 2 | 1.143 | 1.072 | 0.873 | 0.933 | 0.869 | 0.937 |
| 4 | 1.059 | 1.106 | 0.941 | 0.902 | 0.917 | 0.909 |
| 8 | 1.027 | 1.195 | 0.972 | 0.839 | 0.920 | 0.854 |

## Interpretation

This is the first strong KV-cache result in the benchmark. The paired control
shows prefix caching improves throughput and p95 end-to-end latency in both
phase orders. The result contradicts the separate-run Training 022 pilot, which
is exactly why the paired harness matters.

First-event latency remains more order-sensitive. In `cold_first`, the mean
first-event ratio is above `1.0`; in `cache_first`, it is below `1.0`. The
throughput, p95 latency, and TPOT results are more stable and support a
cache-benefit claim for this shared-prefix workload.

The important artifact is not just the speedup. It is the method: same worker,
same scenario plan, paired rows, and phase-order comparison. That is the level
of rigor the KV-cache thread needs if we want research merit rather than a
single noisy benchmark table.

## Next Step

Training 024 stresses the cache mechanism directly: increase shared-prefix
length and add a no-shared-prefix control with matched token counts. The goal is
to separate true prefix reuse from warm-engine and shape effects.

# Training 024: Long Prefix vs Matched Unique-Prefix Control

Training 024 adds two prompt profiles and a profile-control comparison artifact:

- `shared_prefix_long`: long common incident context followed by per-request
  tasks.
- `matched_unique_prefix`: similar topic, size, and output shape, but every
  request starts with a different leading incident identifier so there is no
  large reusable exact prefix.

The paired benchmark runs both profiles in the same randomized scenario plan,
then compares prefix caching on against off under both phase orders.

## Goal

Test whether the Training 023 prefix-cache win is specific to reusable leading
tokens. A stronger result should show `shared_prefix_long` improving more than
`matched_unique_prefix`, especially at higher request counts.

## Commands

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
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 568 \
  --phase-order cache_first \
  --output-dir results/modal-vllm-prefix-cache-long-control-paired-cache-first
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare \
  --prefix-cache-cold-first-paired-dir results/modal-vllm-prefix-cache-long-control-paired \
  --prefix-cache-cache-first-paired-dir results/modal-vllm-prefix-cache-long-control-paired-cache-first \
  --output-dir results/modal-vllm-prefix-cache-long-control-phase-order
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-profile-control \
  --prefix-cache-phase-order-compare-dir results/modal-vllm-prefix-cache-long-control-phase-order \
  --output-dir results/modal-vllm-prefix-cache-long-control-profile-control
```

## Artifacts

```text
results/modal-vllm-prefix-cache-long-control-paired/paired-prefix-cache.json
results/modal-vllm-prefix-cache-long-control-paired/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-long-control-paired/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-long-control-paired-cache-first/paired-prefix-cache.json
results/modal-vllm-prefix-cache-long-control-paired-cache-first/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-long-control-paired-cache-first/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-long-control-phase-order/prefix-cache-phase-order-compare.json
results/modal-vllm-prefix-cache-long-control-phase-order/prefix-cache-phase-order-compare.csv
results/modal-vllm-prefix-cache-long-control-profile-control/prefix-cache-profile-control.json
results/modal-vllm-prefix-cache-long-control-profile-control/prefix-cache-profile-control.csv
```

## Runtime Observations

The prompt profiles are close in shape. In the generated vLLM scenarios,
`shared_prefix_long` averages about `580` prompt tokens and
`matched_unique_prefix` averages about `596` prompt tokens, so the shared-prefix
profile is slightly shorter rather than advantaged by length.

Engine initialization remained heavily order-dependent:

| Phase order | Cold engine load | Cache engine load | Paired runs |
| --- | ---: | ---: | ---: |
| `cold_first` | `150123.545 ms` | `21549.514 ms` | 24 |
| `cache_first` | `42724.541 ms` | `210350.960 ms` | 24 |

## Result

Ratios compare prefix caching on against prefix caching off. Throughput ratios
above `1.0` are better. First-event, latency, and TPOT ratios below `1.0` are
better.

Mean paired-run ratios:

| Phase order | Throughput ratio | First-event ratio | p95 latency ratio | TPOT ratio |
| --- | ---: | ---: | ---: | ---: |
| `cold_first` | 1.090 | 0.851 | 0.932 | 0.958 |
| `cache_first` | 0.989 | 1.399 | 1.044 | 0.978 |

Profile-control mean deltas compare `shared_prefix_long` against
`matched_unique_prefix`. Positive throughput delta means the shared-prefix
profile benefited more. Negative latency or TPOT delta means the shared-prefix
profile benefited more.

| Phase order | Throughput delta | First-event delta | p95 latency delta | TPOT delta |
| --- | ---: | ---: | ---: | ---: |
| `cold_first` | 0.051 | 0.016 | -0.046 | -0.066 |
| `cache_first` | -0.034 | -0.010 | 0.029 | 0.024 |

Throughput ratios by request count:

| Requests | Shared cold-first | Control cold-first | Shared cache-first | Control cache-first |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.013 | 1.117 | 1.057 | 1.054 |
| 2 | 1.095 | 1.030 | 0.969 | 1.041 |
| 4 | 1.080 | 0.967 | 0.986 | 0.929 |
| 8 | 1.158 | 1.029 | 0.980 | 1.104 |

## Interpretation

This is a better experiment than Training 023, but the result is not a clean
prefix-cache win.

In `cold_first`, the long shared-prefix profile behaves the way we hoped:
throughput improves more than the matched unique-prefix control at request
counts `2`, `4`, and `8`, and p95 latency/TPOT also improve more on average.
That supports the hypothesis that longer exact prefixes can expose reusable KV
work.

In `cache_first`, the result flips. The shared-prefix profile no longer beats
the matched control on aggregate throughput or p95 latency, and the request
count `8` control result is stronger than the shared-prefix result. That means
the benchmark still has order sensitivity, warm-state effects, or small-model
cache overhead large enough to contaminate the cache claim.

The value of this artifact is that it prevents overclaiming. Training 024 turns
"prefix caching helped in one paired run" into a sharper statement: the
shared-prefix effect appears under one phase order, but it is not yet stable
under a matched no-prefix control and phase-order inversion.

## Next Step

Training 025 below adds the multi-trial aggregate for the long-control profile
comparison. The next claim should not be "prefix caching is faster"; it should
be "this measured cache-reuse signal survives repeated phase-order controls" or
"it does not."

# Training 025: Long-Control Multi-Trial Aggregate

Training 025 adds a local aggregate mode,
`vllm-prefix-cache-profile-multitrial`, and runs a second independent
long-control trial with `scenario_seed=569`.

## Goal

Check whether the Training 024 shared-vs-control signal survives a second
randomized scenario order. This separates two questions:

- Does prefix caching improve absolute performance in this harness?
- Does the shared-prefix profile benefit more than the matched unique-prefix
  control?

Those are different claims, and Training 025 keeps them separate.

## Commands

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 569 \
  --phase-order cold_first \
  --output-dir results/modal-vllm-prefix-cache-long-control-paired-trial2
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-paired \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 32 \
  --request-counts 1,2,4,8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 569 \
  --phase-order cache_first \
  --output-dir results/modal-vllm-prefix-cache-long-control-paired-cache-first-trial2
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare \
  --prefix-cache-cold-first-paired-dir results/modal-vllm-prefix-cache-long-control-paired-trial2 \
  --prefix-cache-cache-first-paired-dir results/modal-vllm-prefix-cache-long-control-paired-cache-first-trial2 \
  --output-dir results/modal-vllm-prefix-cache-long-control-phase-order-trial2
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-profile-control \
  --prefix-cache-phase-order-compare-dir results/modal-vllm-prefix-cache-long-control-phase-order-trial2 \
  --output-dir results/modal-vllm-prefix-cache-long-control-profile-control-trial2
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-profile-multitrial \
  --prefix-cache-profile-control-dirs results/modal-vllm-prefix-cache-long-control-profile-control,results/modal-vllm-prefix-cache-long-control-profile-control-trial2 \
  --output-dir results/modal-vllm-prefix-cache-long-control-multitrial
```

## Artifacts

```text
results/modal-vllm-prefix-cache-long-control-paired-trial2/paired-prefix-cache.json
results/modal-vllm-prefix-cache-long-control-paired-trial2/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-long-control-paired-trial2/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-long-control-paired-cache-first-trial2/paired-prefix-cache.json
results/modal-vllm-prefix-cache-long-control-paired-cache-first-trial2/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-long-control-paired-cache-first-trial2/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-long-control-phase-order-trial2/prefix-cache-phase-order-compare.json
results/modal-vllm-prefix-cache-long-control-phase-order-trial2/prefix-cache-phase-order-compare.csv
results/modal-vllm-prefix-cache-long-control-profile-control-trial2/prefix-cache-profile-control.json
results/modal-vllm-prefix-cache-long-control-profile-control-trial2/prefix-cache-profile-control.csv
results/modal-vllm-prefix-cache-long-control-multitrial/prefix-cache-profile-multitrial.json
results/modal-vllm-prefix-cache-long-control-multitrial/prefix-cache-profile-multitrial.csv
```

## Runtime Observations

Trial 2 retained the same prompt shape as Training 024. The aggregate reports a
mean shared/control prompt-token ratio of `0.974`, so the shared-prefix profile
remained slightly shorter than the matched unique-prefix control.

Engine initialization remained order-dependent:

| Trial | Phase order | Cold engine load | Cache engine load | Paired runs |
| ---: | --- | ---: | ---: | ---: |
| 1 | `cold_first` | `150123.545 ms` | `21549.514 ms` | 24 |
| 1 | `cache_first` | `42724.541 ms` | `210350.960 ms` | 24 |
| 2 | `cold_first` | `150927.463 ms` | `22173.001 ms` | 24 |
| 2 | `cache_first` | `20187.649 ms` | `142927.172 ms` | 24 |

## Trial 2 Result

Mean paired-run ratios:

| Phase order | Throughput ratio | First-event ratio | p95 latency ratio | TPOT ratio |
| --- | ---: | ---: | ---: | ---: |
| `cold_first` | 0.915 | 1.107 | 1.124 | 1.143 |
| `cache_first` | 0.968 | 1.320 | 1.049 | 1.008 |

Trial 2 does not show an absolute cache-on speedup. Prefix caching was slower on
aggregate in both phase orders.

Trial 2 profile-control mean deltas:

| Phase order | Throughput delta | First-event delta | p95 latency delta | TPOT delta |
| --- | ---: | ---: | ---: | ---: |
| `cold_first` | 0.046 | 0.058 | -0.068 | -0.068 |
| `cache_first` | 0.007 | 0.041 | -0.010 | 0.013 |

## Multi-Trial Result

The two-trial aggregate summarizes shared-minus-control deltas. Positive
throughput means `shared_prefix_long` benefited more than
`matched_unique_prefix`; negative latency/TPOT means `shared_prefix_long`
benefited more.

| Phase order | Metric | Mean delta | Min | Max |
| --- | --- | ---: | ---: | ---: |
| `cold_first` | throughput | 0.048 | 0.046 | 0.051 |
| `cold_first` | first-event | 0.037 | 0.016 | 0.058 |
| `cold_first` | p95 latency | -0.057 | -0.068 | -0.046 |
| `cold_first` | TPOT | -0.067 | -0.068 | -0.066 |
| `cache_first` | throughput | -0.013 | -0.034 | 0.007 |
| `cache_first` | first-event | 0.016 | -0.010 | 0.041 |
| `cache_first` | p95 latency | 0.010 | -0.010 | 0.029 |
| `cache_first` | TPOT | 0.019 | 0.013 | 0.024 |

## Interpretation

Training 025 weakens the absolute prefix-cache speedup claim but preserves a
smaller relative shared-prefix signal.

The absolute cache-on result is not stable. Trial 1 looked favorable in
`cold_first`; Trial 2 was unfavorable in both phase orders. On this small
SmolLM2/T4 harness, prefix caching overhead, warm engine state, JIT behavior,
and scheduler noise are still large enough to swamp the end-to-end result.

The relative profile-control signal is more interesting. In both trials,
`shared_prefix_long` beats `matched_unique_prefix` in `cold_first` throughput by
about five percentage points, and it also has better p95 latency and TPOT
deltas. That suggests the workload construction is detecting some reusable
prefix behavior. The same claim does not hold in `cache_first`, where the mean
throughput delta is slightly negative and the latency/TPOT deltas are worse.

The honest takeaway is: this benchmark can detect a shared-prefix effect under
one controlled phase order, but it cannot yet support a general "prefix caching
improves inference performance" claim.

## Next Step

Training 026 should move from indirect timing evidence to direct cache evidence.
The next useful milestone is to enable or scrape vLLM prefix-cache/KV-cache
metrics, if the installed vLLM version exposes them, and attach hit-rate or
block-reuse counters to each scenario. Without direct cache observability, more
timing trials will mostly quantify noise rather than explain it.

# Training 026: vLLM Cache-Metrics Surface Probe

Training 026 adds `vllm-cache-metrics-probe`, a no-inference Modal introspection
mode for the exact vLLM image used by the benchmark.

## Goal

Determine whether the installed vLLM version exposes cache metrics before we
change the paired benchmark harness. Local Python does not have vLLM installed,
so this probe runs inside the Modal vLLM image and writes a reproducible JSON
artifact.

## Command

```bash
modal run modal_app.py --mode vllm-cache-metrics-probe
```

## Artifact

```text
results/modal-vllm-cache-metrics-probe/cache-metrics-probe.json
```

## Result

The Modal image uses vLLM `0.21.0`. `AsyncEngineArgs` exposes and accepts:

- `kv_cache_metrics=True`
- `kv_cache_metrics_sample=1.0`
- `disable_log_stats=False`

The probe also found these relevant `AsyncLLM` methods:

- `do_log_stats`
- `reset_encoder_cache`
- `reset_mm_cache`
- `reset_prefix_cache`

The no-GPU probe logs that CUDA runtime pieces are not available, but that does
not affect constructor-surface introspection. It does mean the next metrics
experiment should run on the normal GPU-backed vLLM path.

## Interpretation

This is enough to justify a metrics-enabled GPU smoke. vLLM exposes cache metric
flags through the same `AsyncEngineArgs` object we already use, but the Python
object does not obviously expose a direct metrics getter. The likely path is to
enable `kv_cache_metrics`, sample at `1.0`, keep log stats enabled, call
`do_log_stats()` around a tiny shared-prefix run, and capture whatever vLLM
emits.

## Next Step

Training 027 should add a tiny GPU-backed metrics smoke, not a full benchmark:
one shared-prefix scenario, prefix caching enabled, `kv_cache_metrics=True`,
`kv_cache_metrics_sample=1.0`, and explicit `do_log_stats()` calls before and
after the scenario. The artifact should report whether vLLM exposes usable
hit-rate or block-reuse counters in logs or object state.

# Training 027: GPU Cache-Metrics Smoke

Training 027 adds `vllm-cache-metrics-smoke`, a tiny GPU-backed run that enables
vLLM KV-cache metrics and captures the log output from `do_log_stats()`.

## Goal

Move from indirect timing evidence to direct cache observability. The smoke is
intentionally small: four `shared_prefix_long` requests, eight generated tokens,
prefix caching enabled, and metrics sampling set to `1.0`.

## Command

```bash
modal run modal_app.py --mode vllm-cache-metrics-smoke \
  --prompt-profile shared_prefix_long \
  --prompt-count 4 \
  --max-new-tokens 8 \
  --output-dir results/modal-vllm-cache-metrics-smoke
```

## Artifact

```text
results/modal-vllm-cache-metrics-smoke/cache-metrics-smoke.json
```

## Result

The vLLM engine was created with:

- `enable_prefix_caching=True`
- `kv_cache_metrics=True`
- `kv_cache_metrics_sample=1.0`
- `disable_log_stats=False`

The smoke called `do_log_stats()` after engine load and again after the
shared-prefix batch. The second call emitted and captured a vLLM metrics line:

| Metric | Value |
| --- | ---: |
| Prefix cache hit rate | 72.4% |
| Avg prompt throughput | 883.8 tokens/s |
| Avg generation throughput | 44.1 tokens/s |
| GPU KV cache usage after drain | 0.0% |

The request timing summary for the four-request batch was:

| Metric | Value |
| --- | ---: |
| Batch wall time | 726.069 ms |
| Total output tokens | 32 |
| Output tokens/s | 44.073 |
| p95 first chunk | 598.473 ms |
| p95 latency | 725.361 ms |
| p95 TPOT | 20.903 ms |

The Python object surface still does not expose an obvious direct metrics
getter. The visible `AsyncLLM` attributes include `logger_manager`,
`do_log_stats`, `reset_prefix_cache`, and log/cache reset helpers. The usable
metrics path for vLLM `0.21.0` is therefore log-stat capture, not direct object
state.

## Interpretation

This is the first direct cache-observability artifact in the project. The timing
experiments in Trainings 024 and 025 showed mixed end-to-end performance, but
Training 027 confirms that vLLM can report a prefix-cache hit rate for the exact
Modal image and benchmark workload family.

The hit-rate value is not yet a benchmark conclusion by itself. It comes from
one tiny shared-prefix smoke, after the batch has drained. The important result
is methodological: we now know how to enable and capture cache metrics in JSON,
which means the next paired experiments can correlate timing deltas with actual
cache behavior.

## Next Step

Training 028 should wire this log-stat capture into the paired prefix-cache
harness. Each phase should optionally enable `kv_cache_metrics`, call
`do_log_stats()` after warmup and after each scenario or scenario group, parse
prefix-cache hit rate, and attach those metrics to paired rows. Then rerun the
long shared-prefix versus matched unique-prefix control with metrics enabled.

# Training 028: Paired Prefix-Cache Metrics Smoke

Training 028 wires optional vLLM cache-metrics capture into
`vllm-prefix-cache-paired`.

## Goal

Attach direct cache-observability fields to paired benchmark rows. The new
flags are:

- `--cache-metrics on`
- `--kv-cache-metrics-sample 1.0`

When enabled, each engine phase calls `do_log_stats()` after warmup and after
each scenario, captures vLLM log-stat lines, parses prefix-cache hit rate and
KV-cache usage, and writes the metrics into JSON plus summary/run CSVs.

## Command

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
```

## Artifacts

```text
results/modal-vllm-prefix-cache-metrics-paired-smoke/paired-prefix-cache.json
results/modal-vllm-prefix-cache-metrics-paired-smoke/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-metrics-paired-smoke/paired-prefix-cache-runs.csv
```

## Result

This was a tiny smoke, not a full benchmark: two scenarios, one repeat, request
count `4`, output tokens `8`, and `cold_first` phase order.

Mean paired-run ratios:

| Metric | Value |
| --- | ---: |
| Throughput ratio | 1.377 |
| p95 latency ratio | 0.752 |

Scenario rows:

| Profile | Cache hit rate off | Cache hit rate on | Throughput ratio | p95 latency ratio |
| --- | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 0.0% | 58.0% | 1.631 | 0.613 |
| `shared_prefix_long` | 0.0% | 68.1% | 1.122 | 0.892 |

The summary CSV now includes:

- `cold_prefix_cache_hit_rate_pct_median`
- `cache_prefix_cache_hit_rate_pct_median`
- `cache_to_cold_prefix_cache_hit_rate_pct_delta_median`
- `cold_gpu_kv_cache_usage_pct_median`
- `cache_gpu_kv_cache_usage_pct_median`
- `cache_to_cold_gpu_kv_cache_usage_pct_delta_median`

## Interpretation

This is the first paired artifact where timing and direct cache metrics live in
the same rows. It confirms the capture path works inside the paired harness:
cache-disabled rows report `0.0%` prefix-cache hit rate, while cache-enabled
rows report nonzero hit rates.

The hit-rate ordering is sensible for the smoke: `shared_prefix_long` reports a
higher cache hit rate than `matched_unique_prefix`. The timing ordering is not
as simple, because the matched-unique row had a larger throughput and latency
speedup despite a lower hit rate. That is acceptable for this milestone. The
goal was instrumentation, not a final cache-performance claim.

The important change is that future prefix-cache experiments can now reject or
support timing claims using direct cache evidence per scenario.

## Next Step

Training 029 should run the metrics-enabled long-control experiment under both
phase orders with at least the compact grid from Training 028 first, then expand
back to repeats and request counts if the metric fields remain stable. The
phase-order and profile-control comparison artifacts should also carry cache hit
rate deltas forward, not only timing ratios.

# Training 029: Metrics-Aware Phase-Order and Profile-Control Smoke

Training 029 extends the comparison artifacts so cache metrics survive beyond
the raw paired rows.

## Goal

Carry direct cache-hit fields through:

- `vllm-prefix-cache-phase-order-compare`
- `vllm-prefix-cache-profile-control`

This lets later artifacts compare timing and cache behavior in the same table.

## Commands

```bash
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
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare \
  --prefix-cache-cold-first-paired-dir results/modal-vllm-prefix-cache-metrics-paired-smoke \
  --prefix-cache-cache-first-paired-dir results/modal-vllm-prefix-cache-metrics-paired-smoke-cache-first \
  --output-dir results/modal-vllm-prefix-cache-metrics-phase-order-smoke
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-profile-control \
  --prefix-cache-phase-order-compare-dir results/modal-vllm-prefix-cache-metrics-phase-order-smoke \
  --output-dir results/modal-vllm-prefix-cache-metrics-profile-control-smoke
```

## Artifacts

```text
results/modal-vllm-prefix-cache-metrics-paired-smoke-cache-first/paired-prefix-cache.json
results/modal-vllm-prefix-cache-metrics-paired-smoke-cache-first/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-metrics-paired-smoke-cache-first/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-metrics-phase-order-smoke/prefix-cache-phase-order-compare.json
results/modal-vllm-prefix-cache-metrics-phase-order-smoke/prefix-cache-phase-order-compare.csv
results/modal-vllm-prefix-cache-metrics-profile-control-smoke/prefix-cache-profile-control.json
results/modal-vllm-prefix-cache-metrics-profile-control-smoke/prefix-cache-profile-control.csv
```

## Result

This is still the compact smoke grid from Training 028: request count `4`,
output tokens `8`, one repeat, and both long-control prompt profiles.

Phase-order timing and hit-rate rows:

| Profile | Cold-first throughput | Cache-first throughput | Cold-first p95 latency | Cache-first p95 latency | Cache hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 1.631 | 0.593 | 0.613 | 1.687 | 58.0% |
| `shared_prefix_long` | 1.122 | 1.024 | 0.892 | 0.976 | 68.1% |

Profile-control deltas compare `shared_prefix_long` against
`matched_unique_prefix`:

| Phase order | Throughput delta | p95 latency delta | Cache-hit-rate delta |
| --- | ---: | ---: | ---: |
| `cold_first` | -0.509 | 0.279 | 10.1 pp |
| `cache_first` | 0.431 | -0.711 | 10.1 pp |

The metrics-aware comparison artifacts now carry:

- `cold_first_cache_prefix_cache_hit_rate_pct`
- `cache_first_cache_prefix_cache_hit_rate_pct`
- `cold_first_prefix_cache_hit_rate_pct_delta`
- `cache_first_prefix_cache_hit_rate_pct_delta`
- shared-minus-control variants of those fields in profile-control output

## Interpretation

This is the first end-to-end artifact chain where cache hit rate survives from
vLLM logs into paired rows, phase-order rows, and shared-vs-control rows.

The cache metric is stable across phase orders in this compact smoke:
`matched_unique_prefix` reports `58.0%`, while `shared_prefix_long` reports
`68.1%`. The shared profile therefore has a direct cache-hit advantage of about
`10.1` percentage points.

The timing result still depends heavily on phase order. In `cold_first`, the
matched control has the larger throughput win even though it has lower cache hit
rate. In `cache_first`, the shared profile is better on both throughput ratio
and p95 latency ratio. That reinforces the current research position: cache-hit
rate is necessary evidence, but it does not by itself explain end-to-end
latency on this small T4/SmolLM2 harness.

## Next Step

Training 030 should promote this from smoke to a small repeated metric trial:
request counts `2,4,8`, both phase orders, `repeats=2`, and metrics enabled.
The target artifact should answer whether the `~10 pp` shared-prefix hit-rate
advantage is stable and whether any timing metric correlates with that advantage.

# Training 030: Repeated Prefix-Cache Metrics Trial

Training 030 promotes the metrics-aware prefix-cache smoke into a small repeated
trial.

## Goal

Run both phase orders with:

- prompt profiles `shared_prefix_long,matched_unique_prefix`
- output tokens `8`
- request counts `2,4,8`
- `repeats=2`
- `warmup-runs=1`
- `scenario-seed=571`
- vLLM cache metrics enabled

This is still intentionally small. The goal is not a final throughput claim; it
is to test whether direct cache-hit evidence remains visible when we add more
request counts and repeated pairs.

## Commands

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
```

```bash
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
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-phase-order-compare \
  --prefix-cache-cold-first-paired-dir results/modal-vllm-prefix-cache-metrics-repeated \
  --prefix-cache-cache-first-paired-dir results/modal-vllm-prefix-cache-metrics-repeated-cache-first \
  --output-dir results/modal-vllm-prefix-cache-metrics-repeated-phase-order
```

```bash
modal run modal_app.py --mode vllm-prefix-cache-profile-control \
  --prefix-cache-phase-order-compare-dir results/modal-vllm-prefix-cache-metrics-repeated-phase-order \
  --output-dir results/modal-vllm-prefix-cache-metrics-repeated-profile-control
```

## Artifacts

```text
results/modal-vllm-prefix-cache-metrics-repeated/paired-prefix-cache.json
results/modal-vllm-prefix-cache-metrics-repeated/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-metrics-repeated/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-metrics-repeated-cache-first/paired-prefix-cache.json
results/modal-vllm-prefix-cache-metrics-repeated-cache-first/paired-prefix-cache-summary.csv
results/modal-vllm-prefix-cache-metrics-repeated-cache-first/paired-prefix-cache-runs.csv
results/modal-vllm-prefix-cache-metrics-repeated-phase-order/prefix-cache-phase-order-compare.json
results/modal-vllm-prefix-cache-metrics-repeated-phase-order/prefix-cache-phase-order-compare.csv
results/modal-vllm-prefix-cache-metrics-repeated-profile-control/prefix-cache-profile-control.json
results/modal-vllm-prefix-cache-metrics-repeated-profile-control/prefix-cache-profile-control.csv
```

## Result

Each phase-order run produced six scenarios and twelve paired runs.

Top-level paired-run means:

| Phase order | Mean throughput ratio | Mean p95 latency ratio |
| --- | ---: | ---: |
| `cold_first` | 1.204 | 0.957 |
| `cache_first` | 0.778 | 1.760 |

The phase-order comparison rows show the same cache hit-rate values in both
phase orders while timing remains phase-order sensitive:

| Profile | Cold-first throughput | Cache-first throughput | Cold-first p95 latency | Cache-first p95 latency | Cache hit rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` mean | 0.962 | 0.299 | 0.703 | 1.038 | 73.4% |
| `shared_prefix_long` mean | 0.903 | 0.856 | 0.957 | 1.009 | 83.3% |

Shared-prefix minus matched-control deltas by request count:

| Request count | Hit-rate delta | Cold-first throughput delta | Cache-first throughput delta | Cold-first p95 latency delta | Cache-first p95 latency delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 16.4 pp | -0.046 | 0.777 | 0.633 | -0.055 |
| 4 | 9.8 pp | -0.021 | 0.594 | -0.008 | -0.005 |
| 8 | 3.4 pp | -0.111 | 0.298 | 0.136 | -0.027 |
| Mean | 9.9 pp | -0.059 | 0.556 | 0.254 | -0.029 |

## Interpretation

The direct cache evidence survived the move from a one-scenario smoke to a
small repeated trial. `shared_prefix_long` kept a positive cache-hit-rate
advantage over `matched_unique_prefix` at every request count.

The timing story is still not a clean prefix-cache speedup claim. Under
`cold_first`, the shared profile has a higher cache hit rate but slightly worse
mean throughput delta than the matched control. Under `cache_first`, the shared
profile has a large positive throughput-ratio delta and a slightly better p95
latency-ratio delta.

The most important caveat is metric isolation. vLLM `do_log_stats()` reports
engine log-stat values; in this harness those are captured after each scenario,
but they are not proven to be reset per scenario. That means the hit-rate fields
are strong evidence that cache metrics are wired into the artifact chain, and
useful for paired comparisons, but not yet a perfect per-scenario cache-hit
counter.

## Next Step

Training 031 should isolate the cache metric itself. The next benchmark should
run one scenario per fresh engine, or add a counter-delta scrape if vLLM exposes
stable Prometheus counters, so the artifact can distinguish cumulative
engine-level hit rate from per-scenario hit rate. After that, rerun the repeated
profile-control grid with isolated metrics and compare whether the `shared`
advantage still holds.

# Training 031: Fresh-Engine Isolated Cache Metrics

Training 031 adds `vllm-prefix-cache-isolated-metrics`, a local orchestration
mode that reuses the existing remote paired harness but runs one scenario per
fresh remote call.

## Goal

Separate direct cache-hit measurement from cross-scenario carryover. Training
030 showed useful cache metrics, but the matched-unique control still reported
large hit rates. Training 031 removes two sources of contamination:

- each scenario/repeat gets a fresh cold engine and a fresh cache-enabled engine
- warmup scenario runs are forced to `0`

This makes the cache-hit fields more meaningful for the scenario itself. It is
not designed as a final timing benchmark because no-warmup runs expose first
shape and Triton JIT effects.

## Command

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

The mode writes one consolidated artifact after four remote paired calls:

```text
results/modal-vllm-prefix-cache-isolated-metrics/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-metrics/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-isolated-metrics/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-isolated-metrics/prefix-cache-isolated-profile-control.csv
```

## Result

The run produced four scenarios, four paired runs, and four fresh remote calls.

Top-level means:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 2.461 |
| Mean cache-to-cold p95 latency ratio | 0.683 |
| Mean shared-minus-control cache-hit delta | 58.65 pp |

Scenario cache-hit rates:

| Profile | Request count | Cold hit rate | Cache hit rate | Cache-to-cold hit-rate delta |
| --- | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 0.0% | 1.3% | 1.3 pp |
| `matched_unique_prefix` | 4 | 0.0% | 2.0% | 2.0 pp |
| `shared_prefix_long` | 2 | 0.0% | 48.2% | 48.2 pp |
| `shared_prefix_long` | 4 | 0.0% | 72.4% | 72.4 pp |

Shared-prefix minus matched-control deltas:

| Request count | Cache-hit delta | Throughput-ratio delta | p95 latency-ratio delta |
| ---: | ---: | ---: | ---: |
| 2 | 46.9 pp | 5.180 | -0.807 |
| 4 | 70.4 pp | -0.525 | 0.325 |
| Mean | 58.65 pp | 2.328 | -0.241 |

## Interpretation

This is the cleanest cache-observability result so far. With scenario isolation,
the matched-unique control drops to near-zero cache hit rate, while the shared
prefix profile reports large hit rates. That supports the core methodological
claim: the harness can distinguish true reusable-prefix behavior from the
baseline prompt scaffold and prior-scenario carryover.

The result also explains why Training 030 needed caution. Its control-profile
hit rates were much higher because metrics were captured inside a longer-lived
engine sequence. Training 031 shows that a fresh-engine metric run produces a
much sharper shared-vs-control separation.

The timing numbers are useful telemetry, but not yet a fair speedup claim. The
no-warmup isolated run intentionally exposes first-shape/JIT latency. In this
artifact, the n=2 shared-prefix row has a very large throughput ratio partly
because the cold side absorbed a large first-run cost. Future timing claims
should use a separate warmup/discard design.

GPU KV-cache usage still reports `0.0%` because `do_log_stats()` is called after
the batch drains. For this phase of the project, prefix-cache hit rate is the
useful direct metric.

## Next Step

Training 032 should repeat the isolated metric run with request counts `2,4,8`
and `repeats=2`, then separate metric stability from timing stability. If we
want timing evidence, add a dedicated warmup/discard path that logs stats after
warmup and then measures a fresh scenario window without mixing profiles.

# Training 032: Repeated Isolated Cache-Metrics Stability

Training 032 repeats the fresh-engine isolated cache-metrics run across request
counts `2,4,8` with `repeats=2`.

## Goal

Test whether the isolated cache-hit metric from Training 031 is stable across
repeated remote calls and at request count `8`.

This keeps the Training 031 isolation design:

- one scenario per remote paired call
- fresh cold/cache `AsyncLLM` engines per scenario
- `warmup_runs=0`
- cache metrics enabled with `kv_cache_metrics_sample=1.0`

## Command

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

## Artifacts

```text
results/modal-vllm-prefix-cache-isolated-metrics-repeated/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-metrics-repeated/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-isolated-metrics-repeated/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-isolated-metrics-repeated/prefix-cache-isolated-profile-control.csv
```

## Result

The run produced six scenarios, twelve paired runs, and twelve isolated remote
calls.

Top-level means:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.346 |
| Mean cache-to-cold p95 latency ratio | 0.927 |
| Mean shared-minus-control cache-hit delta | 66.5 pp |

Cache-hit stability by scenario:

| Profile | Request count | Cache hit rates | Mean | Population stdev |
| --- | ---: | --- | ---: | ---: |
| `shared_prefix_long` | 2 | `48.2%, 48.2%` | 48.2% | 0.0 |
| `shared_prefix_long` | 4 | `72.4%, 72.4%` | 72.4% | 0.0 |
| `shared_prefix_long` | 8 | `84.6%, 84.6%` | 84.6% | 0.0 |
| `matched_unique_prefix` | 2 | `1.3%, 1.3%` | 1.3% | 0.0 |
| `matched_unique_prefix` | 4 | `2.0%, 2.0%` | 2.0% | 0.0 |
| `matched_unique_prefix` | 8 | `2.4%, 2.4%` | 2.4% | 0.0 |

Shared-prefix minus matched-control deltas:

| Request count | Cache-hit delta | Throughput-ratio delta | p95 latency-ratio delta |
| ---: | ---: | ---: | ---: |
| 2 | 46.9 pp | 0.024 | -0.563 |
| 4 | 70.4 pp | -0.079 | 0.440 |
| 8 | 82.2 pp | 0.071 | -0.135 |
| Mean | 66.5 pp | 0.005 | -0.086 |

## Interpretation

This is now a stable direct-cache result, not just a one-off observation. Under
fresh-engine isolation, the shared-prefix workload reports high prefix-cache hit
rates and the matched-unique control stays near zero. The measured hit rates are
identical across two repeated remote calls for every scenario.

The isolated cache-hit rate also scales with request count for the shared
profile: `48.2%` at two requests, `72.4%` at four requests, and `84.6%` at eight
requests. That is exactly the shape expected when more requests in the same
batch reuse the same long prefix. The matched-control profile rises only from
`1.3%` to `2.4%`, which is consistent with small shared prompt scaffolding rather
than the long reusable prefix.

The timing ratios are much less important here. Their mean deltas are close to
zero in the profile-control table, but no-warmup isolated timing still includes
shape/JIT artifacts. Training 032 should be cited primarily as cache-metric
evidence.

The first remote call saw transient Hugging Face `503` responses while probing
model files, then recovered from the local cache. That affected engine load
time, not the post-scenario prefix-cache hit-rate fields.

## Next Step

Training 033 should turn this into a cleaner report artifact: a compact
machine-readable stability summary with per-scenario hit-rate mean, min, max,
and stdev, plus a markdown/CSV table that can be used directly in the eventual
GitHub project report.

# Training 033: Report-Ready Isolated Stability Summary

Training 033 adds `vllm-prefix-cache-isolated-stability-summary`, a local
artifact generator for the repeated isolated cache-metrics run.

## Goal

Convert the Training 032 run-level artifact into a compact report table with:

- per-scenario cache-hit mean, min, max, and population standard deviation
- shared-prefix versus matched-control deltas
- JSON, CSV, and Markdown outputs

This artifact is meant for the eventual GitHub report. It avoids re-running
GPU inference and reads the existing isolated metrics run CSV.

## Command

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-metrics-repeated \
  --output-dir results/modal-vllm-prefix-cache-isolated-stability-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-isolated-stability-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-isolated-stability-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-isolated-stability-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-isolated-stability-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The summary contains six scenario rows and three profile-control rows.

Top-level summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cache hit rate | 66.500 pp |
| Max cache-hit population stdev | 0.000 |
| Mean shared-minus-control throughput-ratio delta | 0.500 |
| Mean shared-minus-control p95 latency-ratio delta | -0.048 |

Scenario stability:

| Profile | Requests | Runs | Cache hit mean | Min | Max | Stdev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 2 | 1.300% | 1.300% | 1.300% | 0.000 |
| `matched_unique_prefix` | 4 | 2 | 2.000% | 2.000% | 2.000% | 0.000 |
| `matched_unique_prefix` | 8 | 2 | 2.400% | 2.400% | 2.400% | 0.000 |
| `shared_prefix_long` | 2 | 2 | 48.200% | 48.200% | 48.200% | 0.000 |
| `shared_prefix_long` | 4 | 2 | 72.400% | 72.400% | 72.400% | 0.000 |
| `shared_prefix_long` | 8 | 2 | 84.600% | 84.600% | 84.600% | 0.000 |

Shared-prefix versus control:

| Requests | Shared hit mean | Control hit mean | Delta |
| ---: | ---: | ---: | ---: |
| 2 | 48.200% | 1.300% | 46.900 pp |
| 4 | 72.400% | 2.000% | 70.400 pp |
| 8 | 84.600% | 2.400% | 82.200 pp |

## Interpretation

Training 033 turns the core cache-observability result into the first
report-ready table. The main finding is crisp: under fresh-engine isolation,
the shared-prefix workload has a large and repeat-stable prefix-cache hit-rate
advantage, while the matched-unique control remains near zero.

The zero standard deviation is especially useful for the project narrative. It
means the metric path itself is stable for this small harness. We can now use
the isolated metric table as a reference point while improving timing
methodology separately.

The throughput and latency deltas remain secondary. The report artifact carries
them so future work can correlate cache behavior and timing, but the supported
claim at this point is cache reuse, not end-to-end speedup.

## Next Step

Training 034 should improve timing methodology without weakening metric
isolation. The clean next experiment is a two-window design: use one throwaway
shape/JIT warmup window, call `do_log_stats()`, then run the measured scenario
and capture a fresh metric window. If that works, compare isolated cache hit
rate and timing under a fairer warmed-shape condition.

# Training 034: Warm-Window Isolated Cache Trial

Training 034 adds `vllm-prefix-cache-isolated-warm-window`, a sibling of the
fresh-engine isolated metrics mode.

## Goal

Test whether a warmed serving window gives more useful timing evidence while
retaining the one-scenario-per-remote-call isolation from Training 031 and
Training 032.

This mode keeps the same outer isolation:

- one scenario per remote paired call
- fresh cold/cache `AsyncLLM` engines per scenario
- cache metrics enabled with `kv_cache_metrics_sample=1.0`

The difference is that each fresh engine runs one throwaway warmup scenario
before the measured scenario. The paired harness calls `do_log_stats()` after
warmup, then runs the measured scenario and captures the metric log. In the
current harness, the warmup uses the same scenario with `max_tokens_override=1`.

## Command

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

Then generate the compact summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-warm-window \
  --output-dir results/modal-vllm-prefix-cache-isolated-warm-window-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-isolated-warm-window/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-warm-window/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-isolated-warm-window/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-isolated-warm-window/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-isolated-warm-window-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-isolated-warm-window-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-isolated-warm-window-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-isolated-warm-window-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The run produced six scenarios, six paired runs, and six isolated remote calls.

Top-level run means:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.576 |
| Mean cache-to-cold p95 latency ratio | 0.816 |
| Mean shared-minus-control cache-hit delta | 33.233 pp |

Warm-window cache-hit rates:

| Profile | Requests | Cache hit mean |
| --- | ---: | ---: |
| `matched_unique_prefix` | 2 | 50.300% |
| `matched_unique_prefix` | 4 | 50.600% |
| `matched_unique_prefix` | 8 | 50.900% |
| `shared_prefix_long` | 2 | 73.700% |
| `shared_prefix_long` | 4 | 85.800% |
| `shared_prefix_long` | 8 | 92.000% |

Shared-prefix versus control:

| Requests | Cache-hit delta | Throughput-ratio delta | p95 latency-ratio delta |
| ---: | ---: | ---: | ---: |
| 2 | 23.400 pp | 2.605 | -0.700 |
| 4 | 35.200 pp | -0.359 | 0.172 |
| 8 | 41.100 pp | 0.433 | -0.517 |
| Mean | 33.233 pp | 0.893 | -0.348 |

For comparison, the no-warmup isolated stability summary from Training 033 had
a `66.500 pp` mean shared-minus-control cache-hit delta, a `0.500` mean
throughput-ratio delta, and a `-0.048` mean p95 latency-ratio delta.

## Interpretation

Training 034 is useful, but it is not the final timing methodology. The warmup
does improve the aggregate timing story: the mean cache-to-cold throughput ratio
is higher than the no-warmup repeated isolated run, and the mean p95 latency
ratio is lower. The shared-prefix profile also keeps a positive cache-hit
advantage at every request count.

The important caveat is that scenario warmup primes the prefix cache. Because
the warmup uses the same prompts as the measured scenario, the
`matched_unique_prefix` control no longer stays near zero. It rises to about
`50%` cache hit rate in the warmed-window artifact, compared with `1.3%`,
`2.0%`, and `2.4%` in the no-warmup isolated baseline. That cuts the
shared-minus-control cache-hit delta roughly in half.

This means the supported claim is now split:

- Training 032 and Training 033 are the clean direct cache-reuse evidence.
- Training 034 is a warmed-serving timing probe, not a clean profile-control
  cache metric.

Modal stdout also still reported `_compute_slot_mapping_kernel` Triton JIT
warnings during measured inference. A single same-scenario warmup did not fully
remove first-shape effects.

## Next Step

Training 035 should separate shape/JIT warmup from cache warmup. The better
design is a neutral warmup request shape that exercises the same batch size and
decode path without using the measured prompt bodies, or a two-engine design
that warms kernels on neutral prompts and then starts a fresh cache state for
the measured profile. The goal is to keep the stable Training 033 cache metric
separation while improving timing fairness.

# Training 035: Neutral-Warmup Isolated Cache Trial

Training 035 adds `vllm-prefix-cache-isolated-neutral-warmup` and a
`neutral_long` prompt profile.

## Goal

Keep the warmed-serving timing benefits from Training 034 without priming the
measured shared/control prompt bodies. The key change is that warmup uses a
separate long prompt profile:

- measured profiles: `shared_prefix_long`, `matched_unique_prefix`
- warmup profile: `neutral_long`
- warmup runs per fresh engine: `1`
- remote-call isolation: one measured scenario per fresh cold/cache engine

The neutral profile starts each request with a different calibration marker and
uses unrelated long audit text. It is meant to exercise long-prefill shape,
batch size, and decode setup without creating the exact measured prefixes.

## Command

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

Then generate the compact summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup \
  --output-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-isolated-neutral-warmup/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-neutral-warmup/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-isolated-neutral-warmup-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The run produced six scenarios, six paired runs, and six isolated remote calls.

Top-level run means:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.820 |
| Mean cache-to-cold p95 latency ratio | 0.698 |
| Mean shared-minus-control cache-hit delta | 36.667 pp |

Neutral-warmup cache-hit rates:

| Profile | Requests | Cache hit mean |
| --- | ---: | ---: |
| `matched_unique_prefix` | 2 | 2.200% |
| `matched_unique_prefix` | 4 | 2.600% |
| `matched_unique_prefix` | 8 | 2.800% |
| `shared_prefix_long` | 2 | 28.100% |
| `shared_prefix_long` | 4 | 41.400% |
| `shared_prefix_long` | 8 | 48.100% |

Shared-prefix versus control:

| Requests | Cache-hit delta | Throughput-ratio delta | p95 latency-ratio delta |
| ---: | ---: | ---: | ---: |
| 2 | 25.900 pp | 0.148 | -0.016 |
| 4 | 38.800 pp | -0.740 | 0.435 |
| 8 | 45.300 pp | 0.035 | -0.034 |
| Mean | 36.667 pp | -0.186 | 0.129 |

## Comparison

Training 034 same-scenario warmup made the matched-control cache rates jump to
about `50%` for every request count. Training 035 fixes that: the control is
back near the no-warmup isolated baseline.

| Request count | No-warmup control | Same-scenario warmup control | Neutral-warmup control |
| ---: | ---: | ---: | ---: |
| 2 | 1.300% | 50.300% | 2.200% |
| 4 | 2.000% | 50.600% | 2.600% |
| 8 | 2.400% | 50.900% | 2.800% |

The shared-prefix rates are lower than the no-warmup isolated baseline:

| Request count | No-warmup shared | Neutral-warmup shared |
| ---: | ---: | ---: |
| 2 | 48.200% | 28.100% |
| 4 | 72.400% | 41.400% |
| 8 | 84.600% | 48.100% |

## Interpretation

Neutral warmup is the better warmup design. It preserves the matched-control
near-zero cache behavior while producing stronger aggregate cache-to-cold timing
ratios than the no-warmup run.

It also exposes a remaining metric problem: vLLM's logged prefix-cache hit rate
is still cumulative over the warmup and measured scenario. Calling
`do_log_stats()` after warmup does not give this harness a clean measured-only
hit-rate counter. That is why the shared-prefix hit rates are diluted relative
to the no-warmup isolated baseline even though the control stays clean.

Modal stdout continued to report `_compute_slot_mapping_kernel` Triton JIT
warnings during measured inference, so one neutral warmup is still not enough
to fully remove first-shape effects.

## Next Step

Training 036 should compute or capture measured-window cache metrics directly.
There are two promising paths:

- derive an approximate measured-only hit rate from before/after logged rates
  and known prompt-token counts
- find a vLLM counter or metrics endpoint that exposes hit/miss counters, then
  store a true counter delta around the measured scenario

That would let the project combine Training 033's clean cache-reuse result with
Training 035's neutral warmed-serving timing design.

# Training 036: Measured-Window Cache Estimate

Training 036 adds `vllm-prefix-cache-isolated-window-summary`, a local
postprocessor for neutral-warmup isolated artifacts.

## Goal

Estimate the measured-only prefix-cache hit rate from Training 035's
neutral-warmup design. vLLM's logged `Prefix cache hit rate` is cumulative over
the warmup and measured scenario, so the raw after-window value dilutes the
measured workload with neutral warmup tokens.

This step preserves the neutral warmup but records the cache metric snapshot
immediately after warmup. The window summary then subtracts the warmup
contribution from the after-scenario cumulative rate using known prompt-token
counts.

The estimate is intentionally labeled as approximate. It assumes the vLLM
logged rate denominator is proportional to prompt tokens. The stronger future
artifact is still a direct hit/miss counter delta around the measured scenario.

## Command

First rerun the neutral-warmup trial with the enriched warmup metric fields:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long,matched_unique_prefix \
  --output-tokens 8 \
  --request-counts 2,4,8 \
  --repeats 1 \
  --scenario-seed 576 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source
```

Then generate the measured-window estimate:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-window-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source \
  --output-dir results/modal-vllm-prefix-cache-isolated-window-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-window-source/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-isolated-window-summary/prefix-cache-window-summary.json
results/modal-vllm-prefix-cache-isolated-window-summary/prefix-cache-window-summary.csv
results/modal-vllm-prefix-cache-isolated-window-summary/prefix-cache-window-profile-control.csv
results/modal-vllm-prefix-cache-isolated-window-summary/prefix-cache-window-summary.md
```

## Result

The source run produced six scenarios, six paired runs, and six isolated remote
calls. It used `neutral_long` for one warmup run before each measured
cold/cache phase.

Top-level window summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control estimated measured cache hit rate | 66.548 pp |
| Mean shared-minus-control logged after-window cache hit rate | 36.667 pp |

Scenario estimates:

| Profile | Requests | Warmup hit | Logged after | Estimated measured |
| --- | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 1.700% | 2.200% | 2.596% |
| `matched_unique_prefix` | 4 | 2.500% | 2.600% | 2.679% |
| `matched_unique_prefix` | 8 | 3.000% | 2.800% | 2.641% |
| `shared_prefix_long` | 2 | 1.700% | 28.100% | 49.566% |
| `shared_prefix_long` | 4 | 2.500% | 41.400% | 73.113% |
| `shared_prefix_long` | 8 | 3.000% | 48.100% | 84.882% |

Shared-prefix versus matched control:

| Requests | Estimated Shared | Estimated Control | Delta | Throughput Delta | p95 Latency Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 49.566% | 2.596% | 46.970 pp | 2.822 | -0.567 |
| 4 | 73.113% | 2.679% | 70.434 pp | -0.986 | 0.478 |
| 8 | 84.882% | 2.641% | 82.241 pp | 0.074 | -0.039 |

The measured-window estimates line up with the no-warmup shared-prefix rates
from Training 033 while keeping the matched-control workload near zero. That
bridges the metric story across the no-warmup and neutral-warmup experiments:
neutral warmup is not destroying prefix reuse, but the cumulative log field was
masking the measured-window rate.

## Interpretation

Training 036 improves the artifact enough to make the cache-reuse claim more
coherent:

- shared-prefix estimated measured hit rates are high and scale with request
  count: `49.566%`, `73.113%`, `84.882%`
- matched-control estimated measured hit rates remain low:
  `2.596%`, `2.679%`, `2.641%`
- the mean estimated measured shared-minus-control delta is `66.548 pp`, much
  larger than the cumulative after-window delta of `36.667 pp`

The timing deltas remain noisy at this tiny model/request scale. Throughput
delta is strongly positive at two requests, negative at four requests, and only
slightly positive at eight requests. This should still be presented as a
measurement-methodology artifact first, not as a final speedup benchmark.

## Next Step

Training 037 should replace the token-weighted estimate with direct counter
deltas if vLLM exposes cache-hit/cache-query counters in-process or through an
observability endpoint. If that counter is not available, the next best step is
to make the window estimate repeatable across multiple seeds and repeats, then
report confidence intervals for the estimated measured hit-rate delta.

# Training 037: Direct Prefix-Cache Counter Deltas

Training 037 replaces Training 036's token-weighted estimate with direct
measured-window vLLM counter deltas.

## Goal

Capture true prefix-cache query and hit counts around each measured scenario
while keeping Training 035's neutral warmup design.

vLLM's public log line still reports a cumulative hit rate over warmup plus the
measured scenario. The useful object is the per-engine `LoggingStatLogger`
inside `StatLoggerManager`. Its `prefix_caching_metrics` object tracks
`aggregated_query_total` and `aggregated_query_hit`, so the harness now snapshots
those counters before and after each measured scenario and stores the delta.

Implementation references:

- vLLM `StatLoggerManager` and `LoggingStatLogger`: https://docs.vllm.ai/en/v0.13.0/api/vllm/v1/metrics/loggers/
- vLLM `CachingMetrics`: https://docs.vllm.ai/en/v0.14.0/api/vllm/v1/metrics/stats/

## Command

The direct-counter path was first checked with a one-scenario smoke:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_long \
  --output-tokens 8 \
  --request-counts 4 \
  --repeats 1 \
  --scenario-seed 578 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --output-dir results/modal-vllm-prefix-cache-isolated-counter-smoke
```

Then the full comparable grid was rerun:

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

Generate the compact counter-aware summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters \
  --output-dir results/modal-vllm-prefix-cache-isolated-counter-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-isolated-counter-smoke/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counters/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-isolated-counter-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-isolated-counter-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-isolated-counter-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-isolated-counter-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The run produced six scenarios, six paired runs, and six isolated remote calls.

Top-level summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 36.667 pp |
| Mean shared-minus-control direct counter hit rate | 66.501 pp |
| Max direct counter hit-rate population stdev | 0.000 |

Scenario direct counters:

| Profile | Requests | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 2.200% | 2.682% | 1193 | 32 |
| `matched_unique_prefix` | 4 | 2.600% | 2.682% | 2386 | 64 |
| `matched_unique_prefix` | 8 | 2.800% | 2.686% | 4765 | 128 |
| `shared_prefix_long` | 2 | 28.100% | 49.612% | 1161 | 576 |
| `shared_prefix_long` | 4 | 41.400% | 73.040% | 2322 | 1696 |
| `shared_prefix_long` | 8 | 48.100% | 84.901% | 4636 | 3936 |

Shared-prefix versus matched control:

| Requests | Logged Shared | Logged Control | Direct Counter Delta | Throughput Delta | p95 Latency Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 28.100% | 2.200% | 46.930 pp | 2.480 | -0.729 |
| 4 | 41.400% | 2.600% | 70.358 pp | -0.539 | 0.357 |
| 8 | 48.100% | 2.800% | 82.215 pp | 0.048 | -0.065 |

The direct counter deltas match Training 036's estimated measured-window rates
within rounding:

| Requests | Training 036 estimate delta | Training 037 direct delta |
| ---: | ---: | ---: |
| 2 | 46.970 pp | 46.930 pp |
| 4 | 70.434 pp | 70.358 pp |
| 8 | 82.241 pp | 82.215 pp |

## Interpretation

This is the strongest cache-observability result so far. It shows that:

- neutral warmup keeps the matched-control measured-window hit rate low, around
  `2.68%`
- shared-prefix measured-window hit rates are high and scale with request count:
  `49.612%`, `73.040%`, `84.901%`
- the cumulative vLLM log line was the source of dilution, not a loss of prefix
  reuse under neutral warmup

The timing story remains noisy at this small model/request scale. The direct
counter artifact is a measurement-quality improvement first; speedup claims
still need repeated trials and larger workloads.

## Next Step

Training 038 should repeat the direct-counter neutral-warmup design across
multiple seeds and repeats, then report confidence intervals for direct
measured-window cache-hit deltas and timing deltas. The counter path is now good
enough that the next question is stability, not observability.

# Training 038: Direct Counter Stability Intervals

Training 038 repeats the Training 037 direct-counter grid and upgrades the
stability summary with paired-repeat bootstrap intervals.

## Goal

Move from a single direct-counter grid to a repeatable stability artifact.

The summary now pairs shared-prefix and matched-control rows by request shape
and repeat index. It reports distributions and 90% bootstrap intervals for:

- direct measured-window prefix-cache hit-rate delta
- cumulative logged hit-rate delta
- shared-minus-control throughput-ratio delta
- shared-minus-control p95 latency-ratio delta

This keeps the cache-observability claim separate from the timing claim.

## Command

Run the repeated direct-counter neutral-warmup grid:

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
```

Generate the counter-aware stability summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability \
  --output-dir results/modal-vllm-prefix-cache-isolated-counter-stability-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-isolated-neutral-warmup-counter-stability/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-isolated-counter-stability-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-isolated-counter-stability-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-isolated-counter-stability-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-isolated-counter-stability-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The raw run produced six scenarios, eighteen paired runs, and eighteen isolated
remote calls. The top-level raw metrics were:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.186 |
| Mean cache-to-cold latency ratio | 0.909 |
| Mean shared-minus-control logged cache-hit delta | 36.667 pp |
| Mean shared-minus-control direct counter delta | 66.501 pp |

Counter-aware stability summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 36.667 pp |
| Mean shared-minus-control direct counter hit rate | 66.501 pp |
| Direct counter hit-rate 90% bootstrap interval | 58.660 pp to 74.342 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.160 to 0.576 |
| p95 latency-ratio delta 90% bootstrap interval | -0.181 to 0.095 |
| Max direct counter hit-rate population stdev | 0.000 |

Scenario direct counters:

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix` | 2 | 3 | 2.200% | 2.682% | 1193 | 32 |
| `matched_unique_prefix` | 4 | 3 | 2.600% | 2.682% | 2386 | 64 |
| `matched_unique_prefix` | 8 | 3 | 2.800% | 2.686% | 4765 | 128 |
| `shared_prefix_long` | 2 | 3 | 28.100% | 49.612% | 1161 | 576 |
| `shared_prefix_long` | 4 | 3 | 41.400% | 73.040% | 2322 | 1696 |
| `shared_prefix_long` | 8 | 3 | 48.100% | 84.901% | 4636 | 3936 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 46.930 pp | 0.660 | -0.019 to 1.339 | -0.215 | -0.451 to 0.021 |
| 4 | 3 | 70.358 pp | -0.280 | -0.500 to -0.060 | 0.179 | 0.052 to 0.305 |
| 8 | 3 | 82.215 pp | 0.085 | 0.048 to 0.121 | -0.078 | -0.114 to -0.042 |

## Interpretation

The direct-counter cache signal is reproducible in this harness:

- direct shared-minus-control deltas remain `46.930 pp`, `70.358 pp`, and
  `82.215 pp` for request counts 2, 4, and 8
- the mean direct counter delta remains `66.501 pp`, matching Training 037
- matched-control direct measured-window hit rates stay around `2.68%`

The per-shape direct-counter intervals are degenerate because this synthetic
prompt set is deterministic. The aggregate direct-counter interval mainly
reflects the spread across request-count shapes, not instability across fresh
Modal workers.

The timing claim is still not ready. The throughput-ratio interval crosses zero
(`-0.160` to `0.576`), and the p95 latency-ratio interval also crosses zero
(`-0.181` to `0.095`). Per-shape timing remains mixed: request count 2 is
positive for throughput, request count 4 is negative, and request count 8 is
small but positive.

## Next Step

Training 039 should add workload variation instead of only repeating the same
synthetic prompts. The next benchmark should generate multiple shared-prefix
families and matched unique-prefix controls with the same token shapes, then
rerun the direct-counter summary. That will turn the cache-hit interval into a
workload-stability interval rather than a deterministic prompt replay. After
that, the timing path should move to larger request counts or a larger model so
prefill savings are large enough to separate from vLLM startup and scheduler
noise.

# Training 039: Variant Prompt Families

Training 039 adds seed-selected prompt families for the isolated prefix-cache
harness.

## Goal

Make the repeated stability artifact less like deterministic prompt replay.
The new profiles preserve the shared-prefix versus matched-control comparison,
but vary the domain packet by seed:

- `shared_prefix_long_variant` keeps a long identical leading packet within a
  scenario
- `matched_unique_prefix_variant` keeps similar length and topic, but gives each
  request a distinct leading prefix
- isolated repeats pass different scenario seeds, so repeats rotate through
  different workload families

The intent is to keep Training 038's clean direct-counter measurement while
letting the eventual confidence interval reflect workload-family variation.

## Command

Run the variant smoke:

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
```

Generate the variant smoke summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-variant-smoke \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-smoke-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-variant-smoke/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-variant-smoke/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-variant-smoke/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-variant-smoke/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-variant-smoke-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-variant-smoke-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-variant-smoke-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-variant-smoke-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The smoke run produced two scenarios, four paired runs, and four isolated remote
calls.

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 25.600 pp |
| Mean shared-minus-control direct counter hit rate | 63.121 pp |
| Direct counter hit-rate 90% bootstrap interval | 63.048 pp to 63.194 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.167 to -0.091 |
| p95 latency-ratio delta 90% bootstrap interval | 0.097 to 0.252 |
| Max direct counter hit-rate population stdev | 0.079 |

Scenario direct counters:

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_variant` | 4 | 2 | 3.500% | 4.807% | 1331.500 | 64.000 |
| `shared_prefix_long_variant` | 4 | 2 | 29.100% | 67.928% | 1295.500 | 880.000 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2 | 63.121 pp | -0.129 | -0.167 to -0.091 | 0.174 | 0.097 to 0.252 |

## Interpretation

The smoke validates the prompt-family machinery: the shared variant still
produces a large measured-window cache-counter delta, while the matched unique
variant stays low. The timing result is negative in this small smoke, so it
should not be used as an optimization claim.

## Next Step

Run the full variant stability grid across request counts 2, 4, and 8 with
three repeats, then compare its direct-counter interval against Training 038.
If the cache-counter interval stays positive under varied families, the next
research step is to increase model size or request count until timing separates
from startup, scheduler, and small-model decode noise.

# Training 040: Full Variant Stability Grid

Training 040 runs the full variant prompt-family stability grid introduced by
Training 039.

## Goal

Convert the variant prompt-family smoke into a report-ready stability artifact.
This is the first direct-counter interval in the project that varies both
request count and workload family instead of replaying the same synthetic prompt
set.

## Command

Run the full variant grid:

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
```

Generate the full variant stability summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-variant-stability \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-stability-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-variant-stability/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-variant-stability/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-variant-stability/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-variant-stability/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-variant-stability-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-variant-stability-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-variant-stability-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-variant-stability-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The raw run produced six scenarios, eighteen paired runs, and eighteen isolated
remote calls. The top-level raw metrics were:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.063 |
| Mean cache-to-cold latency ratio | 1.018 |
| Mean shared-minus-control logged cache-hit delta | 24.233 pp |
| Mean shared-minus-control direct counter delta | 59.717 pp |

Counter-aware stability summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 24.522 pp |
| Mean shared-minus-control direct counter hit rate | 59.790 pp |
| Direct counter hit-rate 90% bootstrap interval | 52.709 pp to 66.830 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.112 to 0.569 |
| p95 latency-ratio delta 90% bootstrap interval | -0.148 to 0.149 |
| Max direct counter hit-rate population stdev | 0.174 |

Scenario direct counters:

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_variant` | 2 | 3 | 2.967% | 4.727% | 677.333 | 32.000 |
| `matched_unique_prefix_variant` | 4 | 3 | 3.467% | 4.731% | 1353.333 | 64.000 |
| `matched_unique_prefix_variant` | 8 | 3 | 3.700% | 4.739% | 2702.333 | 128.000 |
| `shared_prefix_long_variant` | 2 | 3 | 20.300% | 46.915% | 659.333 | 309.333 |
| `shared_prefix_long_variant` | 4 | 3 | 29.400% | 68.013% | 1317.333 | 896.000 |
| `shared_prefix_long_variant` | 8 | 3 | 34.000% | 78.639% | 2631.333 | 2069.333 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 42.188 pp | 42.065 pp to 42.311 pp | 0.474 | -0.244 to 1.192 | -0.016 | -0.380 to 0.348 |
| 4 | 3 | 63.282 pp | 63.097 pp to 63.467 pp | -0.045 | -0.070 to -0.020 | 0.052 | 0.022 to 0.082 |
| 8 | 3 | 73.899 pp | 73.683 pp to 74.115 pp | 0.000 | -0.015 to 0.016 | -0.002 | -0.018 to 0.015 |

## Interpretation

The cache-reuse signal now has workload-stability evidence. The mean direct
counter delta is `59.790 pp`, and the aggregate 90% interval remains positive
from `52.709 pp` to `66.830 pp`. Per-shape direct-counter deltas also remain
large: `42.188 pp`, `63.282 pp`, and `73.899 pp`.

The timing claim is still not ready. The aggregate throughput interval crosses
zero, and the aggregate p95 latency interval also crosses zero. Request count 4
is actually worse on both throughput and p95 latency in this run, which keeps
the current project claim focused on measured prefix-cache reuse rather than an
end-to-end speedup.

## Next Step

Training 041 should explain the remaining timing ambiguity instead of running
another similar grid. Useful next checks are:

- token and cache-block alignment for each prompt family
- prefill versus decode token accounting by phase
- a larger model or larger request counts so avoided prefill work is large
  enough to compete with scheduler and small-model decode noise

# Training 041: Prompt Block Alignment Audit

Training 041 adds `vllm-prefix-cache-prompt-audit`, a tokenizer-only Modal mode
that estimates exact leading-token overlap and reusable full KV-cache blocks for
the same prompt profiles used by the prefix-cache benchmark.

## Goal

Explain one possible source of timing ambiguity without running another GPU
timing grid. If the variant prompts do not share enough full cache blocks, then
the direct counter result would be fragile and timing should not improve. If the
shared prompts have many reusable full blocks while the controls do not, then
the remaining timing ambiguity is more likely from small-model serving effects,
scheduler noise, decode cost, or phase instrumentation.

## Command

Run the prompt/block audit for the full variant stability grid:

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

## Artifacts

```text
results/modal-vllm-prefix-cache-prompt-audit-variant-stability/prefix-cache-prompt-audit.json
results/modal-vllm-prefix-cache-prompt-audit-variant-stability/prefix-cache-prompt-audit-scenarios.csv
results/modal-vllm-prefix-cache-prompt-audit-variant-stability/prefix-cache-prompt-audit-prompts.csv
results/modal-vllm-prefix-cache-prompt-audit-variant-stability/prefix-cache-prompt-audit-profile-control.csv
results/modal-vllm-prefix-cache-prompt-audit-variant-stability/prefix-cache-prompt-audit.md
```

## Result

The audit used the same model tokenizer, request counts, repeats, and seed as
Training 040. It did not run vLLM inference.

| Metric | Value |
| --- | ---: |
| Scenarios | 18 |
| Prompts | 84 |
| KV cache block size | 16 |
| Mean shared common-prefix full blocks | 18.333 |
| Mean control common-prefix full blocks | 1.000 |
| Mean shared-minus-control common-prefix blocks | 17.333 |
| Mean shared-minus-control reusable block tokens | 1016.889 |
| Mean shared-minus-control reusable block fraction | 0.598 |

Shared versus control block deltas:

| Repeat | Requests | Shared Blocks | Control Blocks | Block Delta | Reusable Token Delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2 | 18 | 1 | 17 | 272 |
| 0 | 4 | 18 | 1 | 17 | 816 |
| 0 | 8 | 18 | 1 | 17 | 1904 |
| 1 | 2 | 18 | 1 | 17 | 272 |
| 1 | 4 | 18 | 1 | 17 | 816 |
| 1 | 8 | 18 | 1 | 17 | 1904 |
| 2 | 2 | 19 | 1 | 18 | 288 |
| 2 | 4 | 19 | 1 | 18 | 864 |
| 2 | 8 | 19 | 1 | 18 | 2016 |

## Interpretation

Block alignment is not the reason timing remains ambiguous. The shared variant
prompts expose 18 to 19 full reusable leading blocks, while the matched unique
controls expose only one full common block from tokenizer/template scaffolding.
This explains why Training 040's direct measured-window counter deltas increase
with request count: the estimated reusable-block-token delta grows from about
`272` to `2016` tokens as the batch grows from two to eight requests.

The timing result is still not ready because block availability alone does not
guarantee an end-to-end win on a 135M model with eight output tokens, fresh
isolated engines, eager execution, and repeated scheduler/JIT effects. Training
041 rules out a prompt-shape failure; it does not prove a serving-speedup claim.

## Next Step

Training 042 should instrument prefill and decode more directly, or rerun one
larger request-count shape to see whether avoided prefill work starts to
dominate small-model overhead. Two useful paths are:

- add a per-phase timing probe that separates prompt prefill time from decode
  time for shared and matched-control profiles
- rerun the variant grid with a larger model or larger request counts so the
  reusable prefill work is big enough to dominate small-model overhead

# Training 042: Variant n=16 Timing Probe

Training 042 reruns the Training 040 variant prompt-family benchmark at a
single larger request count: sixteen requests per measured scenario.

## Goal

Test whether increasing batch size makes the direct measured-window
prefix-cache signal appear in end-to-end timing. Training 040 proved large
shared-prefix counter deltas at request counts two, four, and eight, but the
timing intervals still crossed zero. Training 042 keeps the same model,
profiles, seed, output length, neutral warmup, and cold-first phase order while
doubling the largest previous batch size.

## Command

Run the n=16 variant probe:

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
```

Generate the n=16 stability summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-variant-n16 \
  --prefix-cache-shared-profile shared_prefix_long_variant \
  --prefix-cache-control-profile matched_unique_prefix_variant \
  --output-dir results/modal-vllm-prefix-cache-variant-n16-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-variant-n16/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-variant-n16/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-variant-n16/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-variant-n16/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-variant-n16-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-variant-n16-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-variant-n16-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-variant-n16-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The raw run produced two scenarios, six paired runs, and six isolated remote
calls. The top-level raw metrics were:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.265 |
| Mean cache-to-cold latency ratio | 0.812 |
| Mean shared-minus-control logged cache-hit delta | 15.600 pp |
| Mean shared-minus-control direct counter delta | 38.366 pp |

Counter-aware stability summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 15.700 pp |
| Mean shared-minus-control direct counter hit rate | 38.250 pp |
| Direct counter hit-rate 90% bootstrap interval | 38.162 pp to 38.338 pp |
| Throughput-ratio delta 90% bootstrap interval | 0.108 to 0.422 |
| p95 latency-ratio delta 90% bootstrap interval | -0.212 to -0.082 |
| Max direct counter hit-rate population stdev | 0.114 |

Scenario direct counters:

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_variant` | 16 | 3 | 50.533% | 50.525% | 5404.667 | 2730.667 |
| `shared_prefix_long_variant` | 16 | 3 | 66.233% | 88.775% | 5262.667 | 4672.000 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 38.250 pp | 38.162 pp to 38.338 pp | 0.265 | 0.108 to 0.422 | -0.147 | -0.212 to -0.082 |

## Interpretation

This is the first variant-profile shape where the timing intervals move in the
same direction as the direct counter signal. For request count sixteen, the
shared-prefix workload improved its cache-to-cold throughput ratio more than
the matched-control workload by `0.265`, with a 90% interval from `0.108` to
`0.422`. Its p95 latency ratio also improved by `-0.147`, with a 90% interval
from `-0.212` to `-0.082`.

The result is promising but not yet a general speedup claim. It is one shape,
three paired observations, one small 135M model, eager execution, and a T4. The
matched-control profile also reports a high `50.525%` direct measured-window
hit rate at n=16, which means the control is no longer close to zero-reuse at
this larger batch size. The useful claim is narrower: increasing request count
made the shared-minus-control counter advantage large enough to show up in
timing for this exact configuration.

## Next Step

Training 043 should audit prompt token and cache-block alignment at request
count sixteen. The earlier audit only covered request counts two, four, and
eight. Because the n=16 matched-control profile now reports substantial direct
cache hits, the next check is whether those hits come from repeated full blocks
inside the expanded control batch, tokenizer scaffolding, or a measurement
effect.

# Training 043: n=16 Duplicate Prompt Audit

Training 043 extends `vllm-prefix-cache-prompt-audit` so it reports exact
duplicate prompt groups in addition to the common prefix shared by every prompt
in a scenario. It then reruns the tokenizer-only audit for the n=16 variant
probe from Training 042.

## Goal

Explain why Training 042's matched-control profile reported a high
`50.525%` direct measured-window cache hit rate even though the prompt family
was designed as a unique-prefix control. The prior prompt audit measured only
the leading prefix common to all prompts. At request count sixteen, the variant
prompt generator can also create pairwise exact duplicates because the task
suffix and control label lists each have eight entries.

## Command

Run the n=16 prompt/block audit:

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

## Artifacts

```text
results/modal-vllm-prefix-cache-prompt-audit-variant-n16/prefix-cache-prompt-audit.json
results/modal-vllm-prefix-cache-prompt-audit-variant-n16/prefix-cache-prompt-audit-scenarios.csv
results/modal-vllm-prefix-cache-prompt-audit-variant-n16/prefix-cache-prompt-audit-prompts.csv
results/modal-vllm-prefix-cache-prompt-audit-variant-n16/prefix-cache-prompt-audit-profile-control.csv
results/modal-vllm-prefix-cache-prompt-audit-variant-n16/prefix-cache-prompt-audit.md
```

## Result

The audit used the same model tokenizer, request count, repeats, seed, and
block size as the n=16 timing probe. It did not run vLLM inference.

| Metric | Value |
| --- | ---: |
| Scenarios | 6 |
| Prompts | 96 |
| KV cache block size | 16 |
| Mean shared common-prefix full blocks | 18.333 |
| Mean control common-prefix full blocks | 1.000 |
| Mean shared-minus-control common-prefix blocks | 17.333 |
| Mean shared-minus-control reusable block tokens | 4160.000 |
| Mean shared exact-duplicate reusable block tokens | 2602.667 |
| Mean control exact-duplicate reusable block tokens | 2602.667 |
| Mean shared-minus-control exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block fraction | 0.482 |

Scenario audit:

| Profile | Repeat | Requests | Common Blocks | Unique Prompts | Duplicate Reusable Tokens | Duplicate Fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_prefix_long_variant` | 0 | 16 | 18 | 8 | 2560 | 0.494 |
| `matched_unique_prefix_variant` | 0 | 16 | 1 | 8 | 2560 | 0.481 |
| `shared_prefix_long_variant` | 1 | 16 | 18 | 8 | 2560 | 0.495 |
| `matched_unique_prefix_variant` | 1 | 16 | 1 | 8 | 2560 | 0.482 |
| `shared_prefix_long_variant` | 2 | 16 | 19 | 8 | 2688 | 0.494 |
| `matched_unique_prefix_variant` | 2 | 16 | 1 | 8 | 2688 | 0.482 |

Shared-prefix versus matched control:

| Repeat | Requests | Shared Blocks | Control Blocks | Block Delta | Reusable Token Delta | Shared Duplicate Tokens | Control Duplicate Tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 16 | 18 | 1 | 17 | 4080 | 2560 | 2560 |
| 1 | 16 | 18 | 1 | 17 | 4080 | 2560 | 2560 |
| 2 | 16 | 19 | 1 | 18 | 4320 | 2688 | 2688 |

## Interpretation

Training 043 explains the surprising Training 042 control hit rate. The
matched-control profile still has only one full cache block common to all
prompts, so the original all-prompt common-prefix estimate remains low:
`240` reusable block tokens per repeat, or about `4.3%` to `4.5%` of prompt
tokens. But each n=16 scenario contains only eight unique prompts, with each
exact prompt appearing twice. That duplicate structure contributes about
`2560` to `2688` reusable full-block tokens, or about `48.2%` of the control
prompt tokens. That is consistent with the `50.525%` direct measured-window
control hit rate from Training 042.

This changes the interpretation of Training 042. The timing-positive n=16
result is still useful evidence that larger batches can make cache reuse show
up in timing, but it is not a clean unique-control experiment. Both profiles
benefit from exact duplicate prompts at n=16; the shared profile then adds a
long all-prompt reusable prefix on top.

## Next Step

Training 044 should remove the n=16 duplicate-control confound by adding a
non-cycling variant prompt family or by expanding the task suffix and control
label pools beyond sixteen entries. Then rerun request count sixteen with no
exact duplicate prompts and compare the direct counter and timing intervals
against Training 042.

# Training 044: No-Repeat n=16 Prefix-Cache Probe

Training 044 adds no-repeat variant prompt profiles and reruns the n=16
prefix-cache experiment with the duplicate-control confound removed.

## Goal

Convert Training 042's timing-positive n=16 result into a cleaner
shared-prefix versus unique-prefix comparison. Training 043 showed that the
original n=16 control profile repeated eight exact prompts twice, which
explained its unexpectedly high direct cache hit rate. Training 044 keeps the
same family, seed, request count, output length, neutral warmup, and cold-first
phase order, but uses new prompt profiles that produce sixteen unique prompts
per scenario:

- `shared_prefix_long_no_repeat_variant`
- `matched_unique_prefix_no_repeat_variant`

## Command

Audit the no-repeat prompt/block shape:

```bash
modal run modal_app.py --mode vllm-prefix-cache-prompt-audit \
  --prompt-profiles shared_prefix_long_no_repeat_variant,matched_unique_prefix_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 3 \
  --scenario-seed 577 \
  --kv-cache-block-size 16 \
  --output-dir results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16
```

Run the no-repeat n=16 timing probe:

```bash
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
```

Generate the no-repeat stability summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-n16 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16/prefix-cache-prompt-audit.json
results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16/prefix-cache-prompt-audit-scenarios.csv
results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16/prefix-cache-prompt-audit-prompts.csv
results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16/prefix-cache-prompt-audit-profile-control.csv
results/modal-vllm-prefix-cache-prompt-audit-no-repeat-n16/prefix-cache-prompt-audit.md
results/modal-vllm-prefix-cache-no-repeat-n16/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-no-repeat-n16/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-no-repeat-n16/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-no-repeat-n16/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-n16-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-no-repeat-n16-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-no-repeat-n16-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-n16-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The no-repeat prompt audit confirmed the duplicate confound was removed.

| Metric | Value |
| --- | ---: |
| Scenarios | 6 |
| Prompts | 96 |
| Mean shared common-prefix full blocks | 18.333 |
| Mean control common-prefix full blocks | 1.000 |
| Mean shared-minus-control reusable block tokens | 4160.000 |
| Mean shared exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block tokens | 0.000 |
| Mean control exact-duplicate reusable block fraction | 0.000 |

The raw timing run produced two scenarios, six paired runs, and six isolated
remote calls. The top-level raw metrics were:

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.172 |
| Mean cache-to-cold latency ratio | 0.897 |
| Mean shared-minus-control logged cache-hit delta | 32.400 pp |
| Mean shared-minus-control direct counter delta | 78.395 pp |

Counter-aware stability summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 32.700 pp |
| Mean shared-minus-control direct counter hit rate | 78.472 pp |
| Direct counter hit-rate 90% bootstrap interval | 78.232 pp to 78.713 pp |
| Throughput-ratio delta 90% bootstrap interval | 0.209 to 0.587 |
| p95 latency-ratio delta 90% bootstrap interval | -0.432 to -0.191 |
| Max direct counter hit-rate population stdev | 0.206 |

Scenario direct counters:

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_no_repeat_variant` | 16 | 3 | 31.333% | 4.698% | 5452.333 | 256.000 |
| `shared_prefix_long_no_repeat_variant` | 16 | 3 | 64.033% | 83.170% | 5309.333 | 4416.000 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 3 | 78.472 pp | 78.232 pp to 78.713 pp | 0.398 | 0.209 to 0.587 | -0.311 | -0.432 to -0.191 |

## Interpretation

This is the strongest prefix-cache result in the project so far. With exact
duplicates removed, the no-repeat control drops back to a low direct
measured-window hit rate of `4.698%`, while the shared-prefix profile reports
`83.170%`. The direct shared-minus-control interval is tight and positive:
`78.232 pp` to `78.713 pp`.

The timing result also survives the cleaner control. The shared profile's
cache-to-cold throughput ratio exceeds the control's by `0.398`, with a 90%
interval from `0.209` to `0.587`. The p95 latency ratio improves by `-0.311`,
with a 90% interval from `-0.432` to `-0.191`. This is now a clean n=16
evidence point that exact shared-prefix KV-cache reuse can show up end to end
under this Modal/vLLM/T4 configuration.

The claim still stays scoped. This is one request count, three paired
observations, one small model, eager execution, and one GPU class. It is enough
for a credible benchmark milestone, not enough for a broad serving-system
claim.

## Next Step

Training 045 should test reproducibility rather than add another prompt shape.
The next useful run is a larger no-repeat stability pass, either more repeats
at request count sixteen or a no-repeat request-count sweep such as 8, 12, 16,
and 20. That would tell us whether the timing-positive result is stable across
more paired observations and where the shared-prefix timing effect starts to
separate from small-model noise.

# Training 045: No-Repeat Request-Count Scaling Smoke

Training 045 runs the no-repeat shared-prefix versus matched unique-prefix
profiles across request counts 8, 12, 16, and 20.

## Goal

Check whether the clean Training 044 n=16 result is a one-off or part of a
scaling pattern. This run keeps the duplicate-free prompt profiles, neutral
warmup, cold-first phase order, output length, and seed family, but sweeps
request count with two repeats per profile:

- `shared_prefix_long_no_repeat_variant`
- `matched_unique_prefix_no_repeat_variant`

This is intentionally a smoke, not a final stability pass. The isolated mode
invokes each request count in a separate remote call, which also changes the
vLLM shape configuration for that call: `max_num_batched_tokens` scales with
`request_count * max_model_len`, and `max_num_seqs` equals the request count.
So the result measures a realistic shape-scaling sweep, not a pure request-count
variable with all engine limits fixed.

## Command

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
```

Generate the scaling-smoke stability summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-scaling-smoke \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-scaling-smoke-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-scaling-smoke-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The raw scaling smoke produced eight scenarios, sixteen paired runs, and sixteen
isolated remote calls.

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.097 |
| Mean cache-to-cold latency ratio | 0.969 |
| Mean shared-minus-control logged cache-hit delta | 31.600 pp |
| Mean shared-minus-control direct counter delta | 76.888 pp |

Counter-aware stability summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 31.625 pp |
| Mean shared-minus-control direct counter hit rate | 76.997 pp |
| Direct counter hit-rate 90% bootstrap interval | 75.781 pp to 78.163 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.022 to 0.590 |
| p95 latency-ratio delta 90% bootstrap interval | -0.293 to 0.033 |
| Max direct counter hit-rate population stdev | 0.122 |

Scenario direct counters:

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_no_repeat_variant` | 8 | 2 | 3.700% | 4.815% | 2658.500 | 128.000 |
| `matched_unique_prefix_no_repeat_variant` | 12 | 2 | 22.300% | 4.786% | 4011.500 | 192.000 |
| `matched_unique_prefix_no_repeat_variant` | 16 | 2 | 31.550% | 4.772% | 5364.500 | 256.000 |
| `matched_unique_prefix_no_repeat_variant` | 20 | 2 | 37.100% | 4.769% | 6710.500 | 320.000 |
| `shared_prefix_long_no_repeat_variant` | 8 | 2 | 33.650% | 78.532% | 2587.500 | 2032.000 |
| `shared_prefix_long_no_repeat_variant` | 12 | 2 | 53.800% | 81.547% | 3904.500 | 3184.000 |
| `shared_prefix_long_no_repeat_variant` | 16 | 2 | 63.850% | 83.041% | 5221.500 | 4336.000 |
| `shared_prefix_long_no_repeat_variant` | 20 | 2 | 69.850% | 84.011% | 6532.500 | 5488.000 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 2 | 73.717 pp | 73.617 pp to 73.817 pp | 0.061 | -0.085 to 0.207 | -0.079 | -0.241 to 0.084 |
| 12 | 2 | 76.761 pp | 76.653 pp to 76.869 pp | 0.039 | -0.197 to 0.276 | 0.000 | -0.218 to 0.219 |
| 16 | 2 | 78.269 pp | 78.157 pp to 78.382 pp | 0.680 | -0.126 to 1.487 | -0.273 | -0.656 to 0.110 |
| 20 | 2 | 79.242 pp | 79.127 pp to 79.358 pp | 0.164 | -0.080 to 0.408 | -0.131 | -0.345 to 0.084 |

## Interpretation

The direct prefix-cache counter story is strong and monotonic. The matched
unique-prefix control stays near `4.8%` direct measured-window hit rate at every
request count, while the shared-prefix profile rises from `78.532%` at n=8 to
`84.011%` at n=20. The shared-minus-control direct counter interval is positive
for every request count and the aggregate interval is `75.781 pp` to
`78.163 pp`.

The timing story is weaker than Training 044 because this was only two repeats
per request count. The aggregate throughput delta interval barely crosses zero,
and every per-request-count timing interval crosses zero. The n=16 row has the
largest mean throughput-ratio delta (`0.680`) and the largest mean p95 latency
improvement (`-0.273`), which is consistent with Training 044, but its interval
is wide enough that this run should not be treated as a final timing claim.

Training 045 is useful because it says the no-repeat counter effect generalizes
across request counts. It does not replace the cleaner Training 044 timing
result, which had three n=16 observations and positive timing intervals.

## Next Step

Training 046 should focus on more repeats at a fixed shape, probably n=16. That
is the request count where Training 044 already produced positive timing
intervals and Training 045 produced the largest mean timing separation. A good
next run is eight to ten no-repeat paired observations at n=16, followed by the
same stability summary.

# Training 046: No-Repeat n=16 Fixed-Shape Stability

Training 046 repeats the duplicate-free n=16 experiment eight times at a fixed
shape.

## Goal

Test whether Training 044's positive end-to-end timing result survives more
paired observations when the request count and engine shape stay fixed. This run
uses the same duplicate-free prompt profiles, output length, seed family,
neutral warmup, and cold-first phase order as Training 044:

- `shared_prefix_long_no_repeat_variant`
- `matched_unique_prefix_no_repeat_variant`

Unlike Training 045, this is not a request-count sweep. Every isolated call uses
n=16, so `max_num_batched_tokens` and `max_num_seqs` stay fixed for the measured
shape.

## Command

Run the n=16 fixed-shape stability pass:

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

Generate the stability summary:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary
```

## Artifacts

```text
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-metrics-summary.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-metrics-runs.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8/prefix-cache-isolated-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary/prefix-cache-isolated-stability-summary.md
```

## Result

The raw run produced two scenarios, sixteen paired runs, and sixteen isolated
remote calls.

| Metric | Value |
| --- | ---: |
| Mean cache-to-cold throughput ratio | 1.191 |
| Mean cache-to-cold latency ratio | 0.952 |
| Mean shared-minus-control logged cache-hit delta | 32.400 pp |
| Mean shared-minus-control direct counter delta | 78.395 pp |

Counter-aware stability summary:

| Metric | Value |
| --- | ---: |
| Mean shared-minus-control cumulative logged cache hit rate | 32.600 pp |
| Mean shared-minus-control direct counter hit rate | 78.422 pp |
| Direct counter hit-rate 90% bootstrap interval | 78.269 pp to 78.602 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.332 to 0.185 |
| p95 latency-ratio delta 90% bootstrap interval | -0.180 to 0.023 |
| Max direct counter hit-rate population stdev | 0.197 |

Scenario direct counters:

| Profile | Requests | Runs | Logged Hit Mean | Direct Counter Hit Mean | Direct Queries | Direct Hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `matched_unique_prefix_no_repeat_variant` | 16 | 8 | 31.387% | 4.716% | 5430.375 | 256.000 |
| `shared_prefix_long_no_repeat_variant` | 16 | 8 | 63.987% | 83.138% | 5287.375 | 4396.000 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Direct Counter 90% CI | Throughput Delta | Throughput 90% CI | p95 Latency Delta | p95 Latency 90% CI |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 78.422 pp | 78.269 pp to 78.602 pp | -0.036 | -0.332 to 0.185 | -0.074 | -0.180 to 0.023 |

## Interpretation

The measured-window prefix-cache counter claim is now very strong. Across eight
paired observations, the matched unique-prefix control remains low at `4.716%`
direct hit rate while the shared-prefix profile reaches `83.138%`. The direct
shared-minus-control interval is tight: `78.269 pp` to `78.602 pp`.

The end-to-end timing claim does not survive the larger fixed-shape pass. The
throughput delta mean is slightly negative (`-0.036`) and its 90% interval
crosses zero. The p95 latency delta still leans favorable (`-0.074`), but its
90% interval also crosses zero. Training 044's three-repeat timing-positive
result was therefore too small to treat as stable.

This is still a useful research artifact. It cleanly separates "the cache is
being reused" from "that reuse creates a reliable end-to-end speedup in this
tiny-model Modal/vLLM/T4 setup." The former is proven by direct counters; the
latter is not yet proven by throughput or p95 latency.

## Next Step

Training 047 should improve the stability summary instead of simply adding more
repeats. Prefix caching should primarily affect prefill and TTFT, but the
current profile-control summary only compares throughput and p95 latency. Add
paired bootstrap intervals for first-event/TTFT and stream TPOT ratios, then
rerun the summary on Training 046's existing artifact before launching another
GPU run.

# Training 047: TTFT-Aware Prefix-Cache Summary

Training 047 upgrades the isolated prefix-cache stability summary so it reports
paired bootstrap intervals for prefill-sensitive and decode-sensitive timing
metrics.

## Goal

Training 046 proved the direct cache-counter effect but showed that aggregate
throughput and p95 end-to-end latency did not provide a stable timing claim.
That left a measurement gap: prefix caching should mostly affect prefill and
time-to-first-token, while decode TPOT may not improve. Training 047 fills that
gap without launching another GPU benchmark by adding these profile-control
metrics to the stability summary:

- `shared_minus_control_cache_to_cold_p95_first_event_ratio`
- `shared_minus_control_cache_to_cold_p95_stream_tpot_ratio`

## Command

Regenerate the Training 046 summary with TTFT and TPOT intervals:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8 \
  --prefix-cache-shared-profile shared_prefix_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft
```

## Artifacts

```text
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.json
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-profile-control.csv
results/modal-vllm-prefix-cache-no-repeat-n16-stability-r8-summary-ttft/prefix-cache-isolated-stability-summary.md
```

## Result

The enhanced summary keeps the Training 046 counter and throughput results but
adds first-event/TTFT and stream TPOT intervals.

| Metric | Value |
| --- | ---: |
| Direct counter hit-rate 90% bootstrap interval | 78.269 pp to 78.602 pp |
| Throughput-ratio delta 90% bootstrap interval | -0.332 to 0.185 |
| p95 first-event/TTFT ratio delta 90% bootstrap interval | -0.497 to -0.073 |
| p95 latency-ratio delta 90% bootstrap interval | -0.180 to 0.023 |
| p95 stream TPOT ratio delta 90% bootstrap interval | -0.113 to 0.044 |

Shared-prefix versus matched control:

| Requests | Paired Obs | Direct Counter Delta | Throughput Delta | p95 First-Event Delta | p95 Latency Delta | p95 Stream TPOT Delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 78.422 pp | -0.036 | -0.284 | -0.074 | -0.036 |

## Interpretation

This is the right timing story for the current artifact. Prefix-cache reuse
does not create a stable throughput win in this tiny-model setup, and decode
TPOT remains noisy. But the p95 first-event/TTFT delta is clearly favorable:
the 90% interval is `-0.497` to `-0.073`. That means the cache effect is showing
up where the systems hypothesis predicts it should show up: prefill-sensitive
first-token latency.

The benchmark claim is now more precise:

- The shared-prefix workload reliably triggers measured-window KV-cache reuse.
- The matched unique-prefix control does not.
- The reuse produces a stable first-event/TTFT improvement at n=16.
- It does not yet produce a stable aggregate throughput or decode TPOT claim.

## Next Step

Training 048 should turn this into a report-quality result table and narrative:
counter reuse, TTFT improvement, and non-claims for throughput/TPOT. After that,
the next GPU experiment should stress a more production-relevant setting, such
as a larger model or longer shared prefixes, where prefill dominates more of the
request cost.

# Training 048: Prefix-Cache Study Report

Training 048 packages the current KV-cache evidence into a report-style artifact:

```text
docs/prefix-cache-study.md
```

## Goal

Create a concise GitHub-facing study page that can be read without replaying the
entire Modal training log. The report states the benchmark question,
methodology, primary result table, interpretation, non-claims, reproducibility
commands, and artifact map.

## Result

The report centers the strongest current claim:

- no-repeat shared-prefix prompts produce stable measured-window KV-cache reuse;
- the matched unique-prefix control stays low;
- p95 first-event/TTFT improves for the shared-prefix workload;
- throughput, end-to-end p95 latency, and stream TPOT are not claimed as stable
  wins under the current small-model T4 setup.

## Next Step

Training 049 should either add one more report-quality visualization/table from
the existing CSVs or move to a new GPU experiment that increases prefill cost,
such as longer shared prefixes or a larger model.

# Training 049: Generated Prefix-Cache Result Table

Training 049 adds a small reproducibility helper for the current KV-cache report:

```text
scripts/build_prefix_cache_study_table.py
```

## Goal

Stop hand-copying the headline prefix-cache numbers into report prose. The new
script reads the TTFT-aware stability summary JSON and emits a compact
GitHub-facing table with the direct cache counter effect, first-event/TTFT
effect, and explicit non-claims for throughput, end-to-end p95 latency, and
stream TPOT.

## Result

The generated artifacts are:

```text
results/prefix-cache-study/key-results.md
results/prefix-cache-study/key-results.csv
```

The table preserves the current central result:

- control direct counter hit rate: `4.716%`
- shared direct counter hit rate: `83.138%`
- shared-minus-control direct counter delta: `78.422 pp`, with a 90%
  bootstrap interval from `78.269 pp` to `78.602 pp`
- p95 first-event/TTFT ratio delta: `-0.284`, with a 90% bootstrap interval
  from `-0.497` to `-0.073`
- throughput, end-to-end p95 latency, and stream TPOT intervals still cross
  zero

## Reproduce

```bash
python3 scripts/build_prefix_cache_study_table.py
```

## Next Step

Training 050 should add one visual artifact from the generated table, likely a
small checked-in plot or ASCII-friendly interval chart. After that, move back to
GPU experiments that increase prefill cost with longer prefixes or a larger
model.

# Training 050: Prefix-Cache Interval Chart

Training 050 extends the generated report helper so it emits a lightweight
Markdown interval chart:

```text
results/prefix-cache-study/intervals.md
```

## Goal

Make the report's central distinction visible at a glance. The direct cache
counter interval should be clearly positive, the first-event/TTFT interval
should stay on the favorable side of zero, and the throughput, end-to-end
latency, and stream TPOT intervals should visibly cross zero.

## Result

The chart is generated by:

```bash
python3 scripts/build_prefix_cache_study_table.py
```

It uses only Markdown and fixed-width text, so GitHub renders it without image
dependencies. The important visual result is:

- direct counter delta is tightly positive;
- p95 first-event/TTFT ratio delta stays negative;
- throughput, p95 latency, and stream TPOT ratio intervals cross zero.

## Next Step

Training 051 should return to GPU experimentation. The strongest next run is a
longer-prefix or larger-model variant that increases prefill cost and tests
whether the stable TTFT effect becomes a broader latency or throughput effect.

# Training 051: Extra-Long No-Repeat Prefix-Cache Smoke

Training 051 adds an extra-long prompt/control pair:

```text
shared_prefix_extra_long_no_repeat_variant
matched_unique_prefix_extra_long_no_repeat_variant
neutral_extra_long
```

## Goal

Increase prefill cost while preserving the clean no-repeat control methodology.
The shared profile should have a long identical leading prefix; the matched
control should have nearly the same prompt length but diverge immediately.

## Prompt Audit

Real-tokenizer audit:

```bash
modal run modal_app.py --mode vllm-prefix-cache-prompt-audit \
  --prompt-profiles shared_prefix_extra_long_no_repeat_variant,matched_unique_prefix_extra_long_no_repeat_variant \
  --request-counts 16 \
  --output-tokens 8 \
  --repeats 1 \
  --scenario-seed 577 \
  --kv-cache-block-size 16 \
  --output-dir results/modal-vllm-prefix-cache-prompt-audit-extra-long-no-repeat-n16
```

Result:

- shared mean prompt tokens: about `1,185`
- control mean prompt tokens: about `1,185`
- shared common-prefix full blocks: `72`
- control common-prefix full blocks: `1`
- shared-minus-control reusable block tokens: `17,040`
- exact duplicate reusable tokens: `0` for both profiles

## GPU Smoke

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_extra_long_no_repeat_variant,matched_unique_prefix_extra_long_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 3 \
  --scenario-seed 577 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --warmup-prompt-profile neutral_extra_long \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-smoke-r3
```

Summary artifact:

```text
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-smoke-r3-summary/prefix-cache-isolated-stability-summary.md
```

Key r3 result:

- control direct counter hit rate: `1.342%`
- shared direct counter hit rate: `91.233%`
- shared-minus-control direct counter delta: `89.891 pp`, with a 90%
  bootstrap interval from `89.860 pp` to `89.922 pp`
- p95 first-event/TTFT ratio delta: `-0.618`, interval `-0.671` to `-0.565`
- throughput-ratio delta: `0.834`, interval `0.671` to `0.998`
- p95 latency-ratio delta: `-0.754`, interval `-1.107` to `-0.401`
- p95 stream TPOT-ratio delta: `-0.800`, interval `-1.114` to `-0.485`

Generated follow-up artifacts:

```text
results/prefix-cache-study-extra-long-smoke-r3/key-results.md
results/prefix-cache-study-extra-long-smoke-r3/intervals.md
```

## Caveat

This is a three-repeat smoke. It is strong enough to justify a longer stability
pass, but not enough to replace the existing r8 primary result.

# Training 052: Isolated Metrics Checkpointing

Training 052 fixes a failure mode from the attempted extra-long r8 pass.

## Problem

The r8 run was healthy for multiple remote calls, but Modal stopped the app
before the local entrypoint wrote final artifacts:

```text
ConflictError: function ... is stopped
RemoteError: Function call was cancelled by user or a failure.
```

Because the isolated harness only wrote files at the end, the partial r8 data
was lost.

## Result

The isolated metrics harness now writes checkpoint artifacts after every
completed remote call:

```text
prefix-cache-isolated-metrics.partial.json
prefix-cache-isolated-metrics-summary.partial.csv
prefix-cache-isolated-metrics-runs.partial.csv
prefix-cache-isolated-profile-control.partial.csv
```

The window and stability summary modes prefer final artifacts, but fall back to
partial checkpoint files when the final JSON is missing.

## Next Step

Retry the extra-long stability pass in smaller chunks or move the whole r8
orchestration into a Modal remote function so the local client lifecycle cannot
cancel accumulated work.

# Training 053: Isolated Metrics Chunk Merge

Training 053 adds `vllm-prefix-cache-isolated-merge`, a local artifact combiner
for isolated prefix-cache metrics directories.

## Goal

Make longer stability runs less dependent on one long `modal run` client
lifecycle. Instead of requiring a single r8 run to finish, we can run smaller
chunks, preserve partial checkpoints, and merge completed chunks into the same
file layout that `vllm-prefix-cache-isolated-stability-summary` already reads.

## Result

The merge mode reads final artifacts when present and falls back to checkpoint
artifacts otherwise:

```text
prefix-cache-isolated-metrics.json
prefix-cache-isolated-metrics-runs.csv
prefix-cache-isolated-metrics.partial.json
prefix-cache-isolated-metrics-runs.partial.csv
```

It writes a normal isolated metrics artifact:

```text
prefix-cache-isolated-metrics.json
prefix-cache-isolated-metrics-summary.csv
prefix-cache-isolated-metrics-runs.csv
prefix-cache-isolated-profile-control.csv
prefix-cache-isolated-merge-sources.csv
```

The important implementation detail is repeat reindexing. Every source chunk
starts its local repeats at zero, so the merge assigns global repeat indices
before writing the merged runs CSV. That lets the existing stability summary
keep pairing shared/control observations by repeat without overwriting chunked
observations.

## Reproduce

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-merge \
  --prefix-cache-isolated-merge-dirs results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-smoke-r3,results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r3-seed680 \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6 \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6-summary
```

## Next Step

Run the next extra-long chunk with a new scenario seed, merge it with the r3
smoke, and check whether the broader timing intervals still stay on the
favorable side of zero.

# Training 054: Extra-Long Merged r6 Stability

Training 054 runs the next extra-long r3 chunk with scenario seed `680`, merges
it with the original r3 smoke, and regenerates the stability summary and
GitHub-facing result table.

## Goal

Test whether the broader timing wins from Training 051 survive beyond a
three-repeat smoke. The chunked path uses the merge mode from Training 053 so
the run can be extended incrementally without depending on one long local Modal
entrypoint.

## Reproduce

Run the additional chunk:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_extra_long_no_repeat_variant,matched_unique_prefix_extra_long_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 3 \
  --scenario-seed 680 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --warmup-prompt-profile neutral_extra_long \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r3-seed680
```

Merge and summarize:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-merge \
  --prefix-cache-isolated-merge-dirs results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-smoke-r3,results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r3-seed680 \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6

modal run modal_app.py --mode vllm-prefix-cache-isolated-stability-summary \
  --prefix-cache-isolated-metrics-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6 \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6-summary

python3 scripts/build_prefix_cache_study_table.py \
  --summary-json results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6-summary/prefix-cache-isolated-stability-summary.json \
  --output-dir results/prefix-cache-study-extra-long-merged-r6
```

## Result

The merged artifact has `6` paired shared/control observations for `n=16`.

- control direct counter hit rate: `1.342%`
- shared direct counter hit rate: `91.233%`
- shared-minus-control direct counter delta: `89.891 pp`, interval
  `89.867 pp` to `89.915 pp`
- p95 first-event/TTFT ratio delta: `-0.561`, interval `-0.701` to `-0.376`
- throughput-ratio delta: `0.772`, interval `0.411` to `1.111`
- p95 latency-ratio delta: `-0.525`, interval `-0.867` to `-0.235`
- p95 stream TPOT-ratio delta: `-0.583`, interval `-0.888` to `-0.356`

Generated artifacts:

```text
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r3-seed680/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r6-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-extra-long-merged-r6/key-results.md
results/prefix-cache-study-extra-long-merged-r6/intervals.md
```

## Next Step

The result now supports a stronger extra-long follow-up claim. The next useful
iteration is methodological: either add one more r2 chunk to match the original
eight-repeat count or compare against a larger model/GPU configuration where
prefill cost is more realistic.

# Training 055: Extra-Long Merged r8 Stability

Training 055 adds one more extra-long r2 chunk with scenario seed `790`, then
merges the original r3 smoke, the seed-680 r3 chunk, and the seed-790 r2 chunk
into an eight-observation stability artifact.

## Goal

Match the original no-repeat primary result's repeat count while preserving the
extra-long prompt/control methodology. This gives a cleaner comparison between
the original n=16 no-repeat result and the higher-prefill-cost extra-long
follow-up.

## Reproduce

Run the r2 chunk:

```bash
modal run modal_app.py --mode vllm-prefix-cache-isolated-neutral-warmup \
  --prompt-profiles shared_prefix_extra_long_no_repeat_variant,matched_unique_prefix_extra_long_no_repeat_variant \
  --output-tokens 8 \
  --request-counts 16 \
  --repeats 2 \
  --scenario-seed 790 \
  --phase-order cold_first \
  --kv-cache-metrics-sample 1.0 \
  --warmup-prompt-profile neutral_extra_long \
  --prefix-cache-shared-profile shared_prefix_extra_long_no_repeat_variant \
  --prefix-cache-control-profile matched_unique_prefix_extra_long_no_repeat_variant \
  --output-dir results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r2-seed790
```

Merge and summarize:

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

python3 scripts/build_prefix_cache_study_table.py \
  --summary-json results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.json \
  --output-dir results/prefix-cache-study-extra-long-merged-r8
```

## Result

The merged artifact has `8` paired shared/control observations for `n=16`.

- control direct counter hit rate: `1.343%`
- shared direct counter hit rate: `91.229%`
- shared-minus-control direct counter delta: `89.885 pp`, interval
  `89.867 pp` to `89.909 pp`
- p95 first-event/TTFT ratio delta: `-0.589`, interval `-0.694` to `-0.445`
- throughput-ratio delta: `0.761`, interval `0.489` to `1.011`
- p95 latency-ratio delta: `-0.500`, interval `-0.747` to `-0.284`
- p95 stream TPOT-ratio delta: `-0.567`, interval `-0.791` to `-0.399`

Generated artifacts:

```text
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-chunk-r2-seed790/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8/prefix-cache-isolated-metrics.json
results/modal-vllm-prefix-cache-extra-long-no-repeat-n16-merged-r8-summary/prefix-cache-isolated-stability-summary.md
results/prefix-cache-study-extra-long-merged-r8/key-results.md
results/prefix-cache-study-extra-long-merged-r8/intervals.md
```

## Next Step

The extra-long r8 result is now a credible second result, not just a smoke. The
next research iteration should change hardware/model shape: larger model,
longer context, or a GPU configuration where prefill is a bigger fraction of
serving cost.
