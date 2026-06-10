# GPU 0.30 KV-Budget Feasibility Failure

Command shape:

```bash
modal run modal_app.py --mode vllm-server-async-paired \
  --modal-gpu L4 \
  --hf-model Qwen/Qwen2.5-1.5B-Instruct \
  --prompt-profiles matched_unique_prefix_mega_long_no_repeat_variant \
  --request-counts 32 \
  --output-tokens 8 \
  --repeats 1 \
  --scenario-seed 3906 \
  --warmup-runs 0 \
  --phase-order async_first \
  --server-async-max-num-batched-tokens 60640 \
  --server-async-prefix-caching off \
  --gpu-memory-utilization 0.30 \
  --output-dir results/modal-vllm-server-async-qwen15b-l4-kvbudget-gpu030-n32-batched-tokens60640-matched-unique-seed3906-cache-off-r1
```

Observed failure:

```text
Available KV cache memory: -0.17 GiB
ValueError: No available memory for the cache blocks.
```

Interpretation:

`gpu_memory_utilization=0.30` is below the vLLM startup floor for the current
Qwen2.5-1.5B L4 n32 workload shape. The engine fails during KV-cache
initialization before any paired server/Async measurement artifact is written.
