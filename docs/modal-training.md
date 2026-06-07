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

## Next Modal Step

After the smoke sweep works, add a GPU probe function:

```python
@app.function(gpu="A10")
def gpu_probe() -> dict[str, str | bool]:
    import torch

    return {
        "cuda_available": torch.cuda.is_available(),
        "device": torch.cuda.get_device_name(0),
    }
```

That probe should become the bridge from synthetic simulation to real inference
backend measurements.
