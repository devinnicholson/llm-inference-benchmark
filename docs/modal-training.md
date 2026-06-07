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

Run independent repetitions of both phase orders across fresh Modal workers.
The next benchmark should report confidence intervals for the order effect, not
just one async-first run and one server-first run.
