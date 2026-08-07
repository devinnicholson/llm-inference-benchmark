# Qwen2.5-7B Tensor-Parallel Study on L4

## Status

Complete. The paired vLLM server/`AsyncLLM` harness supports `--modal-gpu L4:2`,
records `tensor_parallel_size=2`, validates the visible GPU count and server
command, and produces matched TP1-versus-TP2 reports. The final aggregate uses
two independent Modal run/seed trials and six paired repeats.

## Question

How do one L4 and two tensor-parallel L4s compare for the same 7B model,
long-context request shape, scheduler budget, and cache-disabled serving path?

This is a topology comparison, not a claim that two GPUs must be faster.
Tensor-parallel communication can outweigh compute savings at modest batch
sizes, so throughput, TTFT, TPOT, KV capacity, and interconnect behavior are all
recorded.

## Protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Topologies: one L4 with TP1 versus two co-located L4s with TP2
- Workload: `shared_prefix_mega_long_no_repeat_variant`, 16 concurrent requests,
  8 output tokens
- Cache: disabled for the HTTP server and in-process `AsyncLLM`
- Scheduler budget: 8,192 batched tokens
- Repeats: one neutral warmup and three measured repeats per trial
- Trials: seeds 4409 and 4509, launched as separate Modal app runs
- Analysis: matched ratios within each repeat; hierarchical bootstrap resampling
  trials and then repeats within trials

The artifacts do not expose physical host identity, so the replication unit is
described as a Modal run/seed trial rather than a host.

## Matched command template

Single L4 baseline:

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --modal-gpu L4 \
  --hf-model Qwen/Qwen2.5-7B-Instruct \
  --request-counts 16 \
  --prompt-profiles shared_prefix_mega_long_no_repeat_variant \
  --output-tokens 8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 4409 \
  --server-async-prefix-caching off \
  --server-async-max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --output-dir results/modal-vllm-qwen7b-l4-tp1-batched8192-r3-seed4409
```

Two-L4 tensor-parallel run:

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --modal-gpu L4:2 \
  --hf-model Qwen/Qwen2.5-7B-Instruct \
  --request-counts 16 \
  --prompt-profiles shared_prefix_mega_long_no_repeat_variant \
  --output-tokens 8 \
  --repeats 3 \
  --warmup-runs 1 \
  --scenario-seed 4409 \
  --server-async-prefix-caching off \
  --server-async-max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.90 \
  --output-dir results/modal-vllm-qwen7b-l4x2-tp2-batched8192-r3-seed4409
```

Repeat both commands with seed 4509, then aggregate with
`scripts/build_tensor_parallel_multitrial.py`.

## Acceptance checks

- Each TP1/TP2 pair reports the same model, request count, prompt profile,
  output length, seed, scheduler budget, cache mode, and effective model length.
- The baseline reports `modal_gpu=L4` and `tensor_parallel_size=1`.
- The multi-GPU artifact reports `modal_gpu=L4:2` and
  `tensor_parallel_size=2`.
- `nvidia_smi_before` and `nvidia_smi_after` show one versus two devices.
- The HTTP server command and in-process engine use the intended
  tensor-parallel size.
- Both execution paths complete all measured repeats before a result is
  included.

## Replicated result

| Metric | TP1 median | TP2 median | TP2 / TP1 | Hierarchical 90% interval |
| --- | ---: | ---: | ---: | ---: |
| Server throughput, output tokens/s | 7.952 | 10.307 | 1.294x | [1.240, 1.361] |
| Server p95 TTFT, ms | 15,645 | 12,218 | 0.780x | [0.741, 0.813] |
| Server p95 latency, ms | 16,099 | 12,465 | 0.774x | [0.735, 0.806] |
| Server p95 TPOT, ms/token | 1,895 | 1,490 | 0.788x | [0.743, 0.824] |

The in-process `AsyncLLM` path independently showed the same direction: 1.276x
throughput and 0.784x p95 latency. Agreement across the HTTP and in-process
paths reduces the chance that the effect is client transport overhead.

vLLM reported 81,784 to 81,785 KV-cache tokens for TP1 and 445,325 to 445,331
for TP2 in both trials, a 5.445x increase. The superlinear increase relative to
device count is consistent with tensor parallelism reducing per-device model
weight pressure and leaving more aggregate memory for KV cache. This is a
systems inference from the topology and vLLM capacity logs, not a separately
instrumented memory decomposition.

The TP2 workers reported no GPU P2P capability, disabled custom all-reduce, and
used NCCL fallback. TP2 still improved latency and throughput, but it was not a
cost-efficiency win: 1.294x throughput for twice the GPUs is 0.647x the TP1
throughput per GPU.

## Feasibility boundary

The first TP1 attempt used the 60,640-token scheduler budget from the smaller
model prefix-cache study. It failed during engine profiling, before any request
timing was collected. For the observed 3,628-token model length, the configured
budget exceeded vLLM's 16-sequence ceiling of 58,048 tokens. The single L4 also
reported 14.29 GiB of model weights and failed a 2.14 GiB allocation with 1.97
GiB free. The matched study therefore uses 8,192 tokens, which allowed both
topologies to initialize and execute the fixed workload with chunked prefill.

See the [failure note](../results/modal-vllm-qwen7b-l4-tp1-batched60640-feasibility-failure-r1/failure.md) and the [replicated aggregate](../results/tensor-parallel-qwen7b-l4-multitrial-r6/tensor-parallel-multitrial.md).

## Interpretation boundary

This experiment supports a narrow claim about a cache-disabled, long-context
Qwen2.5-7B workload on Modal L4s. Two independent trials establish replication,
not fleet-wide variance. The study does not cover production traffic mixtures,
cost-normalized serving, other tensor-parallel degrees, or P2P-capable GPU
interconnects.
