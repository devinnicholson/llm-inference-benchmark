from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import modal


ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = Path("/root")
DEFAULT_WORKLOAD = "generated/mixed_bursty_32_seed568.json"
DEFAULT_MODEL = "llama-7b-gqa-fp16"
DEFAULT_CAPACITIES = "tight-1gb-kv"
DEFAULT_CONCURRENCY = "4"
DEFAULT_POLICIES = "deadline,memory-aware-deadline"
DEFAULT_SWEEP_OUTPUT = "results/modal-training-smoke"
DEFAULT_GPU_PROBE_OUTPUT = "results/modal-gpu-probe"
DEFAULT_INFERENCE_OUTPUT = "results/modal-tiny-inference"
DEFAULT_VLLM_OUTPUT = "results/modal-vllm-inference"
DEFAULT_VLLM_STREAMING_OUTPUT = "results/modal-vllm-streaming"
DEFAULT_VLLM_CONCURRENT_OUTPUT = "results/modal-vllm-concurrent"
DEFAULT_VLLM_SWEEP_OUTPUT = "results/modal-vllm-sweep"
DEFAULT_VLLM_SERVER_OUTPUT = "results/modal-vllm-server-streaming"
DEFAULT_VLLM_SERVER_CONCURRENT_OUTPUT = "results/modal-vllm-server-concurrent"
DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT = "results/modal-vllm-prefix-cache-sweep"
DEFAULT_VLLM_PREFIX_CACHE_COMPARE_OUTPUT = "results/modal-vllm-prefix-cache-compare"
DEFAULT_VLLM_SWEEP_REQUEST_COUNTS = "1,2,4,8"
DEFAULT_VLLM_SWEEP_PROMPT_PROFILES = "short,long"
DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS = "16,32"
DEFAULT_VLLM_SWEEP_REPEATS = 3
DEFAULT_VLLM_SWEEP_SEED = 568
DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_INFERENCE_PROMPT = "Explain KV cache in LLM inference in two concise sentences."
DEFAULT_CONCURRENT_PROMPTS = (
    "Explain KV cache pressure in LLM serving in two concise sentences.",
    "Explain why batching can improve GPU utilization for LLM inference.",
    "Explain the difference between prefill and decode in LLM inference.",
    "Explain why long prompts can hurt tail latency in an inference server.",
)
HF_CACHE_PATH = "/cache"
VLLM_CACHE_PATH = "/vllm-cache"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_dir(ROOT / "src" / "llmbench", remote_path="/root/llmbench")
    .add_local_dir(ROOT / "configs", remote_path="/root/configs")
    .add_local_dir(ROOT / "workloads", remote_path="/root/workloads")
)
gpu_probe_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install("torch", "numpy")
inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("accelerate", "numpy", "safetensors", "torch", "transformers")
    .env({"HF_HOME": HF_CACHE_PATH})
)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "huggingface-hub==0.36.0", "httpx==0.28.1")
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
        }
    )
)
hf_cache_volume = modal.Volume.from_name("llmbench-hf-cache", create_if_missing=True)
vllm_cache_volume = modal.Volume.from_name("llmbench-vllm-cache", create_if_missing=True)
app = modal.App(name="llmbench-modal-training", image=image)


@app.function(timeout=300)
def run_capacity_sweep_remote(
    workload: str = DEFAULT_WORKLOAD,
    model: str = DEFAULT_MODEL,
    capacities: str = DEFAULT_CAPACITIES,
    concurrency: str = DEFAULT_CONCURRENCY,
    policies: str = DEFAULT_POLICIES,
) -> dict[str, Any]:
    from llmbench import run_sweep

    workload_path = REMOTE_ROOT / "workloads" / workload
    model_config_path = _model_config_path(model, root=REMOTE_ROOT)
    capacity_config_paths = [
        _capacity_config_path(capacity, root=REMOTE_ROOT)
        for capacity in _split_csv(capacities)
    ]
    results = run_sweep(
        workload_paths=[workload_path],
        model_config_paths=[model_config_path],
        capacity_config_paths=capacity_config_paths,
        max_concurrent_requests_values=[int(value) for value in _split_csv(concurrency)],
        scheduler_policies=_split_csv(policies),
    )
    return {
        "schema_version": 1,
        "execution": "modal",
        "results": [result.to_record() for result in results],
    }


@app.function(image=gpu_probe_image, gpu="T4", timeout=600)
def run_gpu_probe_remote() -> dict[str, Any]:
    import platform
    import time

    import torch

    nvidia_smi_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    nvidia_smi_full = _run_command(["nvidia-smi"])

    cuda_available = torch.cuda.is_available()
    result: dict[str, Any] = {
        "schema_version": 1,
        "execution": "modal",
        "probe": "gpu",
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "cuda_available": cuda_available,
        "nvidia_smi_query": nvidia_smi_query,
        "nvidia_smi_head": "\n".join(nvidia_smi_full.splitlines()[:8]),
    }
    if not cuda_available:
        return result

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    result.update(
        {
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
            "device_total_memory_mib": properties.total_memory / (1024 * 1024),
            "device_major": properties.major,
            "device_minor": properties.minor,
        }
    )

    x = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    y = torch.randn((1024, 1024), device=device, dtype=torch.float16)
    torch.cuda.synchronize()
    started = time.perf_counter()
    z = x @ y
    torch.cuda.synchronize()
    result.update(
        {
            "matmul_shape": "1024x1024_fp16",
            "matmul_wall_ms": (time.perf_counter() - started) * 1000,
            "matmul_checksum": float(z[0, 0].item()),
        }
    )
    return result


@app.function(
    image=inference_image,
    gpu="T4",
    timeout=900,
    volumes={HF_CACHE_PATH: hf_cache_volume},
)
def run_tiny_inference_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
    warmup_new_tokens: int = 4,
) -> dict[str, Any]:
    import platform

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if warmup_new_tokens <= 0:
        raise ValueError("warmup_new_tokens must be positive")

    nvidia_smi_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_available else "cpu")
    dtype = torch.float16 if cuda_available else torch.float32

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        hf_model,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    model_load_ms = (time.perf_counter() - started) * 1000
    hf_cache_volume.commit()

    tokenization_started = time.perf_counter()
    tokenized, prompt_format = _tokenize_prompt(tokenizer, prompt)
    inputs = {key: value.to(device) for key, value in tokenized.items()}
    tokenization_ms = (time.perf_counter() - tokenization_started) * 1000
    prompt_tokens = int(inputs["input_ids"].shape[-1])

    _synchronize_if_cuda(torch)
    warmup_started = time.perf_counter()
    _manual_greedy_decode(
        model,
        inputs,
        max_new_tokens=warmup_new_tokens,
        torch_module=torch,
    )
    _synchronize_if_cuda(torch)
    warmup_ms = (time.perf_counter() - warmup_started) * 1000

    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    decode_result = _manual_greedy_decode(
        model,
        inputs,
        max_new_tokens=max_new_tokens,
        torch_module=torch,
    )
    generated_ids = decode_result["generated_ids"]
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    generated_tokens = int(generated_ids.shape[-1])
    decode_tokens = max(generated_tokens - 1, 0)
    decode_ms = float(decode_result["decode_ms"])
    ttft_ms = float(decode_result["ttft_ms"])
    total_generation_ms = ttft_ms + decode_ms

    peak_memory_allocated_mib = None
    if cuda_available:
        peak_memory_allocated_mib = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "tiny-inference",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "cpu",
        "cuda_available": bool(cuda_available),
        "nvidia_smi_query": nvidia_smi_query,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "transformers_version": str(transformers.__version__),
        "dtype": str(dtype),
        "tokenizer_load_ms": tokenizer_load_ms,
        "model_load_ms": model_load_ms,
        "tokenization_ms": tokenization_ms,
        "warmup_ms": warmup_ms,
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
        "total_generation_ms": total_generation_ms,
        "tpot_ms": decode_ms / decode_tokens if decode_tokens else 0.0,
        "output_tokens_per_second": generated_tokens / max(total_generation_ms / 1000, 1e-9),
        "peak_memory_allocated_mib": peak_memory_allocated_mib,
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_inference_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import platform
    import time

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
    prompt_format_ms = (time.perf_counter() - started) * 1000
    prompt_tokens = len(tokenizer.encode(formatted_prompt))

    started = time.perf_counter()
    llm = LLM(
        model=hf_model,
        dtype="half",
        max_model_len=1024,
        max_num_batched_tokens=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.50,
        enforce_eager=True,
        trust_remote_code=False,
    )
    engine_load_ms = (time.perf_counter() - started) * 1000

    sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0.0,
    )
    started = time.perf_counter()
    outputs = llm.generate([formatted_prompt], sampling_params, use_tqdm=False)
    generate_wall_ms = (time.perf_counter() - started) * 1000

    output = outputs[0]
    completion = output.outputs[0]
    generated_text = completion.text
    generated_token_ids = list(completion.token_ids)
    generated_tokens = len(generated_token_ids)
    metrics = _serialize_vllm_metrics(output)

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    metric_ttft_ms = _duration_from_metrics_ms(metrics, "arrival_time", "first_token_time")
    metric_total_ms = _duration_from_metrics_ms(metrics, "arrival_time", "finished_time")
    metric_decode_ms = _duration_from_metrics_ms(metrics, "first_token_time", "finished_time")
    decode_tokens = max(generated_tokens - 1, 0)

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-inference",
        "backend": "vllm",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "max_model_len": 1024,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 1,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "engine_load_ms": engine_load_ms,
        "generate_wall_ms": generate_wall_ms,
        "output_tokens_per_second": generated_tokens / max(generate_wall_ms / 1000, 1e-9),
        "metric_ttft_ms": metric_ttft_ms,
        "metric_decode_ms": metric_decode_ms,
        "metric_total_ms": metric_total_ms,
        "metric_tpot_ms": metric_decode_ms / decode_tokens if metric_decode_ms and decode_tokens else None,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "vllm_metrics": metrics,
        "note": "Offline vLLM generate path; streaming TTFT will be measured in a later server-mode run.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_streaming_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import asyncio
    import platform
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
    prompt_format_ms = (time.perf_counter() - started) * 1000
    prompt_tokens = len(tokenizer.encode(formatted_prompt))

    async def run_stream() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=1024,
            max_num_batched_tokens=1024,
            max_num_seqs=1,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            output_kind=RequestOutputKind.DELTA,
        )
        request_id = "modal-vllm-streaming-001"
        first_chunk_ms: float | None = None
        chunks: list[dict[str, Any]] = []
        generated_text_parts: list[str] = []
        generated_tokens = 0
        stream_started = time.perf_counter()
        stream_finished: float | None = None

        try:
            async for output in engine.generate(
                request_id=request_id,
                prompt=formatted_prompt,
                sampling_params=sampling_params,
            ):
                elapsed_ms = (time.perf_counter() - stream_started) * 1000
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                    chunks.append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "text": text,
                            "token_count": len(token_ids),
                            "finished": bool(output.finished),
                        }
                    )
                if output.finished:
                    stream_finished = time.perf_counter()
                    break
            if stream_finished is None:
                stream_finished = time.perf_counter()
        finally:
            engine.shutdown()

        stream_wall_ms = (stream_finished - stream_started) * 1000
        decode_tokens_after_first = max(generated_tokens - 1, 0)
        decode_after_first_ms = (
            stream_wall_ms - first_chunk_ms
            if first_chunk_ms is not None
            else None
        )
        return {
            "engine_load_ms": engine_load_ms,
            "stream_wall_ms": stream_wall_ms,
            "first_chunk_ms": first_chunk_ms,
            "decode_after_first_ms": decode_after_first_ms,
            "stream_tpot_ms": (
                decode_after_first_ms / decode_tokens_after_first
                if decode_after_first_ms is not None and decode_tokens_after_first
                else None
            ),
            "generated_tokens": generated_tokens,
            "generated_text": "".join(generated_text_parts),
            "chunks": chunks,
        }

    stream_result = asyncio.run(run_stream())
    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    stream_wall_ms = float(stream_result["stream_wall_ms"])
    generated_tokens = int(stream_result["generated_tokens"])
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-streaming",
        "backend": "vllm",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "max_model_len": 1024,
        "max_num_batched_tokens": 1024,
        "max_num_seqs": 1,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "engine_load_ms": stream_result["engine_load_ms"],
        "first_chunk_ms": stream_result["first_chunk_ms"],
        "stream_wall_ms": stream_wall_ms,
        "decode_after_first_ms": stream_result["decode_after_first_ms"],
        "stream_tpot_ms": stream_result["stream_tpot_ms"],
        "generated_tokens": generated_tokens,
        "generated_text": stream_result["generated_text"],
        "output_tokens_per_second": generated_tokens / max(stream_wall_ms / 1000, 1e-9),
        "chunks": stream_result["chunks"],
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": "AsyncLLM streaming path using RequestOutputKind.DELTA; first_chunk_ms is measured at first yielded text/token chunk.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_concurrent_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt_count: int = 4,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    import asyncio
    import platform
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if prompt_count <= 0:
        raise ValueError("prompt_count must be positive")

    import vllm

    prompts = _select_concurrent_prompts(prompt_count)
    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    prompt_records = []
    prompt_format_started = time.perf_counter()
    for index, prompt in enumerate(prompts):
        formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
        prompt_records.append(
            {
                "request_id": f"concurrent-{index:02d}",
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "prompt_format": prompt_format,
                "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
            }
        )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    async def run_concurrent() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=1024,
            max_num_batched_tokens=2048,
            max_num_seqs=prompt_count,
            gpu_memory_utilization=0.50,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0.0,
            output_kind=RequestOutputKind.DELTA,
        )

        async def stream_one(record: dict[str, Any], batch_started: float) -> dict[str, Any]:
            request_started = time.perf_counter()
            first_chunk_ms: float | None = None
            finished_ms: float | None = None
            chunks = []
            generated_text_parts = []
            generated_tokens = 0
            async for output in engine.generate(
                request_id=record["request_id"],
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                batch_elapsed_ms = (now - batch_started) * 1000
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                    chunks.append(
                        {
                            "elapsed_ms": elapsed_ms,
                            "batch_elapsed_ms": batch_elapsed_ms,
                            "token_count": len(token_ids),
                            "text": text,
                            "finished": bool(output.finished),
                        }
                    )
                if output.finished:
                    finished_ms = elapsed_ms
                    break

            stream_wall_ms = finished_ms or ((time.perf_counter() - request_started) * 1000)
            decode_tokens_after_first = max(generated_tokens - 1, 0)
            decode_after_first_ms = (
                stream_wall_ms - first_chunk_ms
                if first_chunk_ms is not None
                else None
            )
            return {
                "request_id": record["request_id"],
                "prompt": record["prompt"],
                "prompt_tokens": record["prompt_tokens"],
                "first_chunk_ms": first_chunk_ms,
                "stream_wall_ms": stream_wall_ms,
                "decode_after_first_ms": decode_after_first_ms,
                "stream_tpot_ms": (
                    decode_after_first_ms / decode_tokens_after_first
                    if decode_after_first_ms is not None and decode_tokens_after_first
                    else None
                ),
                "generated_tokens": generated_tokens,
                "generated_text": "".join(generated_text_parts),
                "chunks": chunks,
            }

        batch_started = time.perf_counter()
        try:
            request_results = await asyncio.gather(
                *(stream_one(record, batch_started) for record in prompt_records)
            )
        finally:
            engine.shutdown()
        batch_wall_ms = (time.perf_counter() - batch_started) * 1000
        return {
            "engine_load_ms": engine_load_ms,
            "batch_wall_ms": batch_wall_ms,
            "requests": request_results,
        }

    batch_result = asyncio.run(run_concurrent())
    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    requests = batch_result["requests"]
    total_output_tokens = sum(request["generated_tokens"] for request in requests)
    first_chunks = [
        request["first_chunk_ms"]
        for request in requests
        if request["first_chunk_ms"] is not None
    ]
    latencies = [request["stream_wall_ms"] for request in requests]
    tpot_values = [
        request["stream_tpot_ms"]
        for request in requests
        if request["stream_tpot_ms"] is not None
    ]
    batch_wall_ms = float(batch_result["batch_wall_ms"])
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-concurrent",
        "backend": "vllm",
        "model_id": hf_model,
        "request_count": len(requests),
        "max_new_tokens": int(max_new_tokens),
        "max_model_len": 1024,
        "max_num_batched_tokens": 2048,
        "max_num_seqs": prompt_count,
        "prompt_format_ms": prompt_format_ms,
        "tokenizer_load_ms": tokenizer_load_ms,
        "engine_load_ms": batch_result["engine_load_ms"],
        "batch_wall_ms": batch_wall_ms,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
        "p50_first_chunk_ms": _percentile(first_chunks, 50),
        "p95_first_chunk_ms": _percentile(first_chunks, 95),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p50_stream_tpot_ms": _percentile(tpot_values, 50),
        "p95_stream_tpot_ms": _percentile(tpot_values, 95),
        "requests": requests,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": "Concurrent AsyncLLM streaming workload. Request timings are measured from each coroutine start after the shared engine is loaded.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_sweep_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    enable_prefix_caching: bool = False,
) -> dict[str, Any]:
    import asyncio
    import platform
    import random
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [profile.lower() for profile in _split_csv(prompt_profiles)]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    for profile in prompt_profile_values:
        if profile not in {"short", "long", "mixed"}:
            raise ValueError("prompt_profiles must contain only short, long, or mixed")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    scenario_specs = []
    prompt_format_started = time.perf_counter()
    for prompt_profile in prompt_profile_values:
        for max_new_tokens in output_token_values:
            for request_count in request_count_values:
                prompt_records = []
                prompts = _select_sweep_prompts(request_count, prompt_profile)
                for index, prompt in enumerate(prompts):
                    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
                    prompt_records.append(
                        {
                            "request_id": (
                                f"{prompt_profile}-out{max_new_tokens}-n{request_count}-"
                                f"{index:02d}"
                            ),
                            "prompt": prompt,
                            "formatted_prompt": formatted_prompt,
                            "prompt_format": prompt_format,
                            "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                        }
                    )
                prompt_tokens = [record["prompt_tokens"] for record in prompt_records]
                scenario_specs.append(
                    {
                        "scenario_id": f"{prompt_profile}_out{max_new_tokens}_n{request_count}",
                        "prompt_profile": prompt_profile,
                        "request_count": request_count,
                        "max_new_tokens": max_new_tokens,
                        "prompt_format": prompt_records[0]["prompt_format"],
                        "prompt_tokens_min": min(prompt_tokens),
                        "prompt_tokens_max": max(prompt_tokens),
                        "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
                        "total_prompt_tokens": sum(prompt_tokens),
                        "prompt_records": prompt_records,
                    }
                )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_output_tokens = max(output_token_values)
    max_prompt_tokens = max(spec["prompt_tokens_max"] for spec in scenario_specs)
    max_model_len = max(1024, max_prompt_tokens + max_output_tokens + 32)
    max_num_batched_tokens = max(2048, max_request_count * max_model_len)

    async def run_streaming_sweep() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_request_count,
            gpu_memory_utilization=0.50,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000

        async def stream_one(
            scenario_id: str,
            run_id: str,
            record: dict[str, Any],
            sampling_params: Any,
            batch_started: float,
        ) -> dict[str, Any]:
            request_started = time.perf_counter()
            first_chunk_ms: float | None = None
            finished_ms: float | None = None
            generated_text_parts = []
            generated_tokens = 0
            chunk_count = 0

            async for output in engine.generate(
                request_id=f"{run_id}-{scenario_id}-{record['request_id']}",
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                chunk_count += 1
                for completion in output.outputs:
                    text = completion.text
                    token_ids = list(completion.token_ids or [])
                    if first_chunk_ms is None and (text or token_ids):
                        first_chunk_ms = elapsed_ms
                    if text:
                        generated_text_parts.append(text)
                    generated_tokens += len(token_ids)
                if output.finished:
                    finished_ms = elapsed_ms
                    break

            stream_wall_ms = finished_ms or ((time.perf_counter() - request_started) * 1000)
            decode_tokens_after_first = max(generated_tokens - 1, 0)
            decode_after_first_ms = (
                stream_wall_ms - first_chunk_ms
                if first_chunk_ms is not None
                else None
            )
            generated_text = "".join(generated_text_parts)
            return {
                "request_id": record["request_id"],
                "prompt_tokens": record["prompt_tokens"],
                "first_chunk_ms": first_chunk_ms,
                "stream_wall_ms": stream_wall_ms,
                "decode_after_first_ms": decode_after_first_ms,
                "stream_tpot_ms": (
                    decode_after_first_ms / decode_tokens_after_first
                    if decode_after_first_ms is not None and decode_tokens_after_first
                    else None
                ),
                "generated_tokens": generated_tokens,
                "total_sequence_tokens": record["prompt_tokens"] + generated_tokens,
                "chunk_count": chunk_count,
                "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
                "generated_text": generated_text,
            }

        async def drain_warmup_request(
            scenario_id: str,
            record: dict[str, Any],
            sampling_params: Any,
        ) -> int:
            generated_tokens = 0
            async for output in engine.generate(
                request_id=f"warmup-{scenario_id}-{record['request_id']}",
                prompt=record["formatted_prompt"],
                sampling_params=sampling_params,
            ):
                for completion in output.outputs:
                    generated_tokens += len(list(completion.token_ids or []))
                if output.finished:
                    break
            return generated_tokens

        async def run_warmup() -> dict[str, Any]:
            sampling_params = SamplingParams(
                max_tokens=1,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            started = time.perf_counter()
            generated_tokens = 0
            for spec in scenario_specs:
                warmup_counts = await asyncio.gather(
                    *(
                        drain_warmup_request(
                            spec["scenario_id"],
                            record,
                            sampling_params,
                        )
                        for record in spec["prompt_records"]
                    )
                )
                generated_tokens += sum(warmup_counts)
            return {
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "shape_warmup_scenarios": len(scenario_specs),
                "generated_tokens": generated_tokens,
            }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
        ) -> dict[str, Any]:
            sampling_params = SamplingParams(
                max_tokens=spec["max_new_tokens"],
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(
                    stream_one(
                        spec["scenario_id"],
                        run_id,
                        record,
                        sampling_params,
                        batch_started,
                    )
                    for record in spec["prompt_records"]
                )
            )
            batch_wall_ms = (time.perf_counter() - batch_started) * 1000
            summary = _summarize_stream_requests(request_results, batch_wall_ms)
            return {
                "run_id": run_id,
                "repeat_index": repeat_index,
                "run_order": run_order,
                "scenario_id": spec["scenario_id"],
                "prompt_profile": spec["prompt_profile"],
                "request_count": spec["request_count"],
                "max_new_tokens": spec["max_new_tokens"],
                "prompt_format": spec["prompt_format"],
                "prompt_tokens_min": spec["prompt_tokens_min"],
                "prompt_tokens_max": spec["prompt_tokens_max"],
                "prompt_tokens_mean": spec["prompt_tokens_mean"],
                "total_prompt_tokens": spec["total_prompt_tokens"],
                "estimated_peak_sequence_tokens": sum(
                    request["total_sequence_tokens"] for request in request_results
                ),
                **summary,
                "requests": request_results,
            }

        try:
            warmup_result = await run_warmup()
            scenario_plan = []
            for repeat_index in range(repeats):
                for spec in scenario_specs:
                    scenario_plan.append((repeat_index, spec))
            random.Random(scenario_seed).shuffle(scenario_plan)
            scenario_runs = []
            for run_order, (repeat_index, spec) in enumerate(scenario_plan):
                scenario_runs.append(
                    await run_scenario(
                        spec,
                        repeat_index=repeat_index,
                        run_order=run_order,
                    )
                )
        finally:
            engine.shutdown()

        return {
            "engine_load_ms": engine_load_ms,
            "warmup": warmup_result,
            "scenario_runs": scenario_runs,
        }

    sweep_result = asyncio.run(run_streaming_sweep())
    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    scenario_runs = sweep_result["scenario_runs"]
    scenarios = _aggregate_vllm_sweep_scenarios(scenario_specs, scenario_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-sweep",
        "backend": "vllm",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "scenario_count": len(scenarios),
        "scenario_run_count": len(scenario_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "enable_prefix_caching": bool(enable_prefix_caching),
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "engine_load_ms": sweep_result["engine_load_ms"],
        "warmup": sweep_result["warmup"],
        "summary": _vllm_sweep_summary_rows(scenarios),
        "run_summary": _vllm_sweep_run_rows(scenario_runs),
        "scenarios": scenarios,
        "scenario_runs": scenario_runs,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": (
            "One loaded AsyncLLM engine runs a seeded, repeated, shuffled grid "
            "over request count, prompt profile, and output token budget after "
            f"one-token shape warmups with prefix caching {bool(enable_prefix_caching)}."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_streaming_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    max_new_tokens: int = 32,
    ready_timeout_s: int = 600,
) -> dict[str, Any]:
    import json
    import platform
    import subprocess
    import threading
    import time

    import httpx
    from transformers import AutoTokenizer

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
    prompt_tokens = len(tokenizer.encode(formatted_prompt))

    port = 8000
    base_url = f"http://127.0.0.1:{port}"
    command = [
        "vllm",
        "serve",
        hf_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "half",
        "--max-model-len",
        "1024",
        "--max-num-batched-tokens",
        "1024",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
    ]

    server_logs: list[str] = []
    server_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def drain_logs() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            server_logs.append(line.rstrip())

    log_thread = threading.Thread(target=drain_logs, daemon=True)
    log_thread.start()

    ready_ms: float | None = None
    usage: dict[str, Any] | None = None
    response_status_code: int | None = None
    chunks: list[dict[str, Any]] = []
    generated_text_parts: list[str] = []

    try:
        with httpx.Client(timeout=2.0) as health_client:
            while (time.perf_counter() - server_started) < ready_timeout_s:
                if process.poll() is not None:
                    raise RuntimeError(
                        "vLLM server exited before readiness: "
                        + "\n".join(_tail_lines(server_logs, 40))
                    )
                try:
                    response = health_client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        ready_ms = (time.perf_counter() - server_started) * 1000
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1.0)
        if ready_ms is None:
            raise TimeoutError(
                "vLLM server did not become ready: "
                + "\n".join(_tail_lines(server_logs, 40))
            )

        request_body = {
            "model": hf_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first_content_ms: float | None = None
        request_started = time.perf_counter()
        with httpx.Client(timeout=None) as client:
            with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response_status_code = response.status_code
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    choices = event.get("choices", [])
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if content and first_content_ms is None:
                            first_content_ms = elapsed_ms
                        if content:
                            generated_text_parts.append(content)
                        chunks.append(
                            {
                                "elapsed_ms": elapsed_ms,
                                "content": content,
                                "finish_reason": choice.get("finish_reason"),
                            }
                        )
        request_wall_ms = (time.perf_counter() - request_started) * 1000
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        log_thread.join(timeout=5)

    generated_text = "".join(generated_text_parts)
    generated_tokens = (
        int(usage["completion_tokens"])
        if usage and usage.get("completion_tokens") is not None
        else len(tokenizer.encode(generated_text, add_special_tokens=False))
    )
    measured_prompt_tokens = (
        int(usage["prompt_tokens"])
        if usage and usage.get("prompt_tokens") is not None
        else prompt_tokens
    )
    decode_tokens_after_first = max(generated_tokens - 1, 0)
    decode_after_first_ms = (
        request_wall_ms - first_content_ms
        if first_content_ms is not None
        else None
    )

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-streaming",
        "backend": "vllm-openai-server",
        "model_id": hf_model,
        "prompt": prompt,
        "prompt_format": prompt_format,
        "prompt_tokens": measured_prompt_tokens,
        "prompt_tokens_local_estimate": prompt_tokens,
        "max_new_tokens": int(max_new_tokens),
        "generated_tokens": generated_tokens,
        "generated_text": generated_text,
        "response_status_code": response_status_code,
        "server_ready_ms": ready_ms,
        "request_wall_ms": request_wall_ms,
        "first_content_ms": first_content_ms,
        "decode_after_first_ms": decode_after_first_ms,
        "stream_tpot_ms": (
            decode_after_first_ms / decode_tokens_after_first
            if decode_after_first_ms is not None and decode_tokens_after_first
            else None
        ),
        "output_tokens_per_second": generated_tokens / max(request_wall_ms / 1000, 1e-9),
        "chunk_count": len(chunks),
        "chunks": chunks,
        "usage": usage,
        "server_command": command,
        "tokenizer_load_ms": tokenizer_load_ms,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_logs[:80],
        "server_logs_tail": _tail_lines(server_logs, 80),
        "note": "OpenAI-compatible vLLM server smoke using /v1/chat/completions with SSE streaming.",
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_concurrent_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profile: str = "short",
    max_new_tokens: int = 32,
    ready_timeout_s: int = 600,
) -> dict[str, Any]:
    import asyncio
    import json
    import platform
    import subprocess
    import threading
    import time

    import httpx
    from transformers import AutoTokenizer

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile = prompt_profile.lower()
    if prompt_profile not in {"short", "long", "mixed"}:
        raise ValueError("prompt_profile must be short, long, or mixed")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")

    import vllm

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    tokenizer_load_ms = (time.perf_counter() - started) * 1000

    scenario_specs = []
    prompt_format_started = time.perf_counter()
    for request_count in request_count_values:
        prompt_records = []
        for index, prompt in enumerate(_select_sweep_prompts(request_count, prompt_profile)):
            formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
            prompt_records.append(
                {
                    "request_id": f"{prompt_profile}-n{request_count}-{index:02d}",
                    "prompt": prompt,
                    "formatted_prompt": formatted_prompt,
                    "prompt_format": prompt_format,
                    "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                }
            )
        prompt_tokens = [record["prompt_tokens"] for record in prompt_records]
        scenario_specs.append(
            {
                "scenario_id": f"{prompt_profile}_out{max_new_tokens}_n{request_count}",
                "prompt_profile": prompt_profile,
                "request_count": request_count,
                "max_new_tokens": max_new_tokens,
                "prompt_format": prompt_records[0]["prompt_format"],
                "prompt_tokens_min": min(prompt_tokens),
                "prompt_tokens_max": max(prompt_tokens),
                "prompt_tokens_mean": sum(prompt_tokens) / len(prompt_tokens),
                "total_prompt_tokens": sum(prompt_tokens),
                "prompt_records": prompt_records,
            }
        )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_model_len = 1024
    max_num_batched_tokens = max(2048, max_request_count * max_model_len)
    port = 8000
    base_url = f"http://127.0.0.1:{port}"
    command = [
        "vllm",
        "serve",
        hf_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "half",
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--max-num-seqs",
        str(max_request_count),
        "--gpu-memory-utilization",
        "0.50",
        "--enforce-eager",
    ]

    server_logs: list[str] = []
    server_started = time.perf_counter()
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def drain_logs() -> None:
        if process.stdout is None:
            return
        for line in process.stdout:
            server_logs.append(line.rstrip())

    log_thread = threading.Thread(target=drain_logs, daemon=True)
    log_thread.start()

    async def stream_one(
        scenario_id: str,
        record: dict[str, Any],
        batch_started: float,
    ) -> dict[str, Any]:
        request_body = {
            "model": hf_model,
            "messages": [{"role": "user", "content": record["prompt"]}],
            "temperature": 0.0,
            "max_tokens": max_new_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        chunks: list[dict[str, Any]] = []
        generated_text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        response_status_code: int | None = None
        first_content_ms: float | None = None
        request_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json=request_body,
                headers={"Accept": "text/event-stream"},
            ) as response:
                response_status_code = response.status_code
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[len("data: ") :]
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    batch_elapsed_ms = (time.perf_counter() - batch_started) * 1000
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if content and first_content_ms is None:
                            first_content_ms = elapsed_ms
                        if content:
                            generated_text_parts.append(content)
                        chunks.append(
                            {
                                "elapsed_ms": elapsed_ms,
                                "batch_elapsed_ms": batch_elapsed_ms,
                                "content": content,
                                "finish_reason": choice.get("finish_reason"),
                            }
                        )
        request_wall_ms = (time.perf_counter() - request_started) * 1000
        generated_text = "".join(generated_text_parts)
        generated_tokens = (
            int(usage["completion_tokens"])
            if usage and usage.get("completion_tokens") is not None
            else len(tokenizer.encode(generated_text, add_special_tokens=False))
        )
        prompt_tokens = (
            int(usage["prompt_tokens"])
            if usage and usage.get("prompt_tokens") is not None
            else record["prompt_tokens"]
        )
        decode_tokens_after_first = max(generated_tokens - 1, 0)
        decode_after_first_ms = (
            request_wall_ms - first_content_ms
            if first_content_ms is not None
            else None
        )
        return {
            "request_id": record["request_id"],
            "scenario_id": scenario_id,
            "prompt": record["prompt"],
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_local_estimate": record["prompt_tokens"],
            "generated_tokens": generated_tokens,
            "total_sequence_tokens": prompt_tokens + generated_tokens,
            "generated_text": generated_text,
            "response_status_code": response_status_code,
            "first_content_ms": first_content_ms,
            "request_wall_ms": request_wall_ms,
            "decode_after_first_ms": decode_after_first_ms,
            "stream_tpot_ms": (
                decode_after_first_ms / decode_tokens_after_first
                if decode_after_first_ms is not None and decode_tokens_after_first
                else None
            ),
            "chunk_count": len(chunks),
            "chunks": chunks,
            "usage": usage,
        }

    async def run_scenario(spec: dict[str, Any]) -> dict[str, Any]:
        batch_started = time.perf_counter()
        request_results = await asyncio.gather(
            *(
                stream_one(
                    spec["scenario_id"],
                    record,
                    batch_started,
                )
                for record in spec["prompt_records"]
            )
        )
        batch_wall_ms = (time.perf_counter() - batch_started) * 1000
        total_output_tokens = sum(request["generated_tokens"] for request in request_results)
        first_contents = [
            request["first_content_ms"]
            for request in request_results
            if request["first_content_ms"] is not None
        ]
        latencies = [request["request_wall_ms"] for request in request_results]
        tpot_values = [
            request["stream_tpot_ms"]
            for request in request_results
            if request["stream_tpot_ms"] is not None
        ]
        return {
            "scenario_id": spec["scenario_id"],
            "prompt_profile": spec["prompt_profile"],
            "request_count": spec["request_count"],
            "max_new_tokens": spec["max_new_tokens"],
            "prompt_format": spec["prompt_format"],
            "prompt_tokens_min": spec["prompt_tokens_min"],
            "prompt_tokens_max": spec["prompt_tokens_max"],
            "prompt_tokens_mean": spec["prompt_tokens_mean"],
            "total_prompt_tokens": spec["total_prompt_tokens"],
            "estimated_peak_sequence_tokens": sum(
                request["total_sequence_tokens"] for request in request_results
            ),
            "batch_wall_ms": batch_wall_ms,
            "total_output_tokens": total_output_tokens,
            "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
            "p50_first_content_ms": _percentile(first_contents, 50),
            "p95_first_content_ms": _percentile(first_contents, 95),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "p50_stream_tpot_ms": _percentile(tpot_values, 50),
            "p95_stream_tpot_ms": _percentile(tpot_values, 95),
            "requests": request_results,
        }

    ready_ms: float | None = None
    scenarios: list[dict[str, Any]] = []
    try:
        with httpx.Client(timeout=2.0) as health_client:
            while (time.perf_counter() - server_started) < ready_timeout_s:
                if process.poll() is not None:
                    raise RuntimeError(
                        "vLLM server exited before readiness: "
                        + "\n".join(_tail_lines(server_logs, 40))
                    )
                try:
                    response = health_client.get(f"{base_url}/health")
                    if response.status_code == 200:
                        ready_ms = (time.perf_counter() - server_started) * 1000
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(1.0)
        if ready_ms is None:
            raise TimeoutError(
                "vLLM server did not become ready: "
                + "\n".join(_tail_lines(server_logs, 40))
            )

        for spec in scenario_specs:
            scenarios.append(asyncio.run(run_scenario(spec)))
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)
        log_thread.join(timeout=5)

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-concurrent",
        "backend": "vllm-openai-server",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profile": prompt_profile,
        "max_new_tokens": int(max_new_tokens),
        "scenario_count": len(scenarios),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "server_ready_ms": ready_ms,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "summary": _vllm_server_concurrent_summary_rows(scenarios),
        "scenarios": scenarios,
        "server_command": command,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_logs[:80],
        "server_logs_tail": _tail_lines(server_logs, 80),
        "note": "OpenAI-compatible vLLM server concurrent streaming workload using /v1/chat/completions.",
    }


@app.local_entrypoint()
def main(
    mode: str = "sweep",
    workload: str = DEFAULT_WORKLOAD,
    model: str = DEFAULT_MODEL,
    hf_model: str = DEFAULT_HF_MODEL,
    prompt: str = DEFAULT_INFERENCE_PROMPT,
    capacities: str = DEFAULT_CAPACITIES,
    concurrency: str = DEFAULT_CONCURRENCY,
    policies: str = DEFAULT_POLICIES,
    max_new_tokens: int = 32,
    prompt_count: int = 4,
    prompt_profile: str = "short",
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    prefix_caching: str = "off",
    cold_sweep_dir: str = DEFAULT_VLLM_SWEEP_OUTPUT,
    prefix_sweep_dir: str = DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT,
    output_dir: str = "",
) -> None:
    if mode == "gpu-probe":
        payload = run_gpu_probe_remote.remote()
        output_path = Path(output_dir or DEFAULT_GPU_PROBE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "probe.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"cuda_available: {payload['cuda_available']}")
        if payload.get("device_name"):
            print(f"device: {payload['device_name']}")
        print(f"json: {json_path}")
        return

    if mode == "tiny-inference":
        payload = run_tiny_inference_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_INFERENCE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "inference.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"device: {payload['device_name']}")
        print(f"ttft_ms: {payload['ttft_ms']:.3f}")
        print(f"tpot_ms: {payload['tpot_ms']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-inference":
        payload = run_vllm_inference_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-inference.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"generate_wall_ms: {payload['generate_wall_ms']:.3f}")
        print(f"output_tokens_per_second: {payload['output_tokens_per_second']:.3f}")
        if payload.get("metric_ttft_ms") is not None:
            print(f"metric_ttft_ms: {payload['metric_ttft_ms']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-streaming":
        payload = run_vllm_streaming_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_STREAMING_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-streaming.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        if payload.get("first_chunk_ms") is not None:
            print(f"first_chunk_ms: {payload['first_chunk_ms']:.3f}")
        if payload.get("stream_tpot_ms") is not None:
            print(f"stream_tpot_ms: {payload['stream_tpot_ms']:.3f}")
        print(f"output_tokens_per_second: {payload['output_tokens_per_second']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-concurrent":
        payload = run_vllm_concurrent_remote.remote(
            hf_model=hf_model,
            prompt_count=prompt_count,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_CONCURRENT_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-concurrent.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"requests: {payload['request_count']}")
        print(f"batch_wall_ms: {payload['batch_wall_ms']:.3f}")
        print(f"p95_first_chunk_ms: {payload['p95_first_chunk_ms']:.3f}")
        print(f"aggregate_output_tokens_per_second: {payload['aggregate_output_tokens_per_second']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-server-streaming":
        payload = run_vllm_server_streaming_remote.remote(
            hf_model=hf_model,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-server-streaming.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"server_ready_ms: {payload['server_ready_ms']:.3f}")
        if payload.get("first_content_ms") is not None:
            print(f"first_content_ms: {payload['first_content_ms']:.3f}")
        if payload.get("stream_tpot_ms") is not None:
            print(f"stream_tpot_ms: {payload['stream_tpot_ms']:.3f}")
        print(f"output_tokens_per_second: {payload['output_tokens_per_second']:.3f}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-server-concurrent":
        payload = run_vllm_server_concurrent_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profile=prompt_profile,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_CONCURRENT_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-server-concurrent.json"
        csv_path = output_path / "vllm-server-concurrent.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"server_ready_ms: {payload['server_ready_ms']:.3f}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"max_num_seqs: {payload['max_num_seqs']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-sweep":
        enable_prefix_caching = _parse_bool_choice(prefix_caching, "prefix_caching")
        payload = run_vllm_sweep_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            enable_prefix_caching=enable_prefix_caching,
        )
        default_output = (
            DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT
            if enable_prefix_caching
            else DEFAULT_VLLM_SWEEP_OUTPUT
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-sweep.json"
        csv_path = output_path / "vllm-sweep.csv"
        run_csv_path = output_path / "vllm-sweep-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["run_summary"])

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"scenario_runs: {payload['scenario_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"scenario_seed: {payload['scenario_seed']}")
        print(f"enable_prefix_caching: {payload['enable_prefix_caching']}")
        print(f"max_num_seqs: {payload['max_num_seqs']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"runs_csv: {run_csv_path}")
        return

    if mode == "vllm-prefix-cache-compare":
        payload = _compare_vllm_sweep_csvs(
            cold_csv=Path(cold_sweep_dir) / "vllm-sweep.csv",
            prefix_csv=Path(prefix_sweep_dir) / "vllm-sweep.csv",
            cold_label="prefix_caching_off",
            prefix_label="prefix_caching_on",
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-compare.json"
        csv_path = output_path / "prefix-cache-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode != "sweep":
        raise ValueError(
            "mode must be 'sweep', 'gpu-probe', 'tiny-inference', "
            "'vllm-inference', 'vllm-streaming', 'vllm-concurrent', "
            "'vllm-server-streaming', 'vllm-server-concurrent', "
            "'vllm-sweep', or "
            "'vllm-prefix-cache-compare'"
        )

    payload = run_capacity_sweep_remote.remote(
        workload=workload,
        model=model,
        capacities=capacities,
        concurrency=concurrency,
        policies=policies,
    )
    output_path = Path(output_dir or DEFAULT_SWEEP_OUTPUT)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / "sweep-results.json"
    csv_path = output_path / "sweep-results.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_records_csv(csv_path, payload["results"])

    print(f"cases: {len(payload['results'])}")
    print(f"json: {json_path}")
    print(f"csv: {csv_path}")


def _split_csv(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("comma-separated argument must contain at least one value")
    return values


def _split_positive_int_csv(value: str, label: str) -> list[int]:
    raw_values = _split_csv(value)
    parsed_values = []
    for raw_value in raw_values:
        try:
            parsed_value = int(raw_value)
        except ValueError as error:
            raise ValueError(f"{label} must contain positive integers") from error
        if parsed_value <= 0:
            raise ValueError(f"{label} must contain positive integers")
        parsed_values.append(parsed_value)
    return parsed_values


def _parse_bool_choice(value: str, label: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    raise ValueError(f"{label} must be on or off")


def _select_concurrent_prompts(prompt_count: int) -> list[str]:
    prompts = []
    for index in range(prompt_count):
        base_prompt = DEFAULT_CONCURRENT_PROMPTS[index % len(DEFAULT_CONCURRENT_PROMPTS)]
        if index < len(DEFAULT_CONCURRENT_PROMPTS):
            prompts.append(base_prompt)
        else:
            prompts.append(f"{base_prompt} Variation {index + 1}.")
    return prompts


def _select_sweep_prompts(prompt_count: int, prompt_profile: str) -> list[str]:
    prompt_profile = prompt_profile.lower()
    short_prompts = _select_concurrent_prompts(prompt_count)
    if prompt_profile == "short":
        return short_prompts
    long_prompts = [
        _extend_prompt_for_context(prompt, index)
        for index, prompt in enumerate(short_prompts)
    ]
    if prompt_profile == "long":
        return long_prompts
    if prompt_profile == "mixed":
        return [
            prompt if index % 2 == 0 else long_prompts[index]
            for index, prompt in enumerate(short_prompts)
        ]
    raise ValueError("prompt_profile must be short, long, or mixed")


def _extend_prompt_for_context(prompt: str, index: int) -> str:
    return (
        f"{prompt} Consider request group {index + 1} in a bursty production "
        "serving workload with mixed prompt lengths, streaming responses, and a "
        "fixed GPU memory budget. Compare prefill cost, decode cost, scheduler "
        "queueing, KV-cache growth, cache block fragmentation, and tail latency. "
        "Use one concrete systems example and identify the bottleneck that would "
        "matter most on a memory-constrained GPU."
    )


def _summarize_stream_requests(
    requests: list[dict[str, Any]],
    batch_wall_ms: float,
) -> dict[str, Any]:
    total_output_tokens = sum(request["generated_tokens"] for request in requests)
    first_chunks = [
        request["first_chunk_ms"]
        for request in requests
        if request["first_chunk_ms"] is not None
    ]
    latencies = [request["stream_wall_ms"] for request in requests]
    tpot_values = [
        request["stream_tpot_ms"]
        for request in requests
        if request["stream_tpot_ms"] is not None
    ]
    return {
        "batch_wall_ms": batch_wall_ms,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
        "p50_first_chunk_ms": _percentile(first_chunks, 50),
        "p95_first_chunk_ms": _percentile(first_chunks, 95),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p50_stream_tpot_ms": _percentile(tpot_values, 50),
        "p95_stream_tpot_ms": _percentile(tpot_values, 95),
    }


def _aggregate_vllm_sweep_scenarios(
    scenario_specs: list[dict[str, Any]],
    scenario_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    runs_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for run in scenario_runs:
        runs_by_scenario.setdefault(run["scenario_id"], []).append(run)

    scenarios = []
    for spec in scenario_specs:
        runs = sorted(
            runs_by_scenario.get(spec["scenario_id"], []),
            key=lambda run: (run["repeat_index"], run["run_order"]),
        )
        scenario: dict[str, Any] = {
            "scenario_id": spec["scenario_id"],
            "prompt_profile": spec["prompt_profile"],
            "request_count": spec["request_count"],
            "max_new_tokens": spec["max_new_tokens"],
            "prompt_format": spec["prompt_format"],
            "prompt_tokens_min": spec["prompt_tokens_min"],
            "prompt_tokens_max": spec["prompt_tokens_max"],
            "prompt_tokens_mean": spec["prompt_tokens_mean"],
            "total_prompt_tokens": spec["total_prompt_tokens"],
            "repeats": len(runs),
            "repeat_indices": [run["repeat_index"] for run in runs],
            "run_orders": [run["run_order"] for run in runs],
        }
        for field in (
            "estimated_peak_sequence_tokens",
            "batch_wall_ms",
            "total_output_tokens",
            "aggregate_output_tokens_per_second",
            "p50_first_chunk_ms",
            "p95_first_chunk_ms",
            "p50_latency_ms",
            "p95_latency_ms",
            "p50_stream_tpot_ms",
            "p95_stream_tpot_ms",
        ):
            scenario.update(_metric_distribution(runs, field))
        scenarios.append(scenario)
    return scenarios


def _vllm_sweep_summary_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "repeats",
        "estimated_peak_sequence_tokens_median",
        "estimated_peak_sequence_tokens_p95",
        "batch_wall_ms_median",
        "batch_wall_ms_p95",
        "batch_wall_ms_min",
        "batch_wall_ms_max",
        "batch_wall_ms_cv",
        "total_output_tokens_median",
        "aggregate_output_tokens_per_second_median",
        "aggregate_output_tokens_per_second_p95",
        "aggregate_output_tokens_per_second_min",
        "aggregate_output_tokens_per_second_max",
        "aggregate_output_tokens_per_second_cv",
        "p95_first_chunk_ms_median",
        "p95_first_chunk_ms_p95",
        "p95_first_chunk_ms_min",
        "p95_first_chunk_ms_max",
        "p95_first_chunk_ms_cv",
        "p95_latency_ms_median",
        "p95_latency_ms_p95",
        "p95_latency_ms_min",
        "p95_latency_ms_max",
        "p95_latency_ms_cv",
        "p95_stream_tpot_ms_median",
        "p95_stream_tpot_ms_p95",
        "p95_stream_tpot_ms_min",
        "p95_stream_tpot_ms_max",
        "p95_stream_tpot_ms_cv",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in scenarios
    ]


def _vllm_sweep_run_rows(scenario_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    row_fields = (
        "run_id",
        "repeat_index",
        "run_order",
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "estimated_peak_sequence_tokens",
        "batch_wall_ms",
        "total_output_tokens",
        "aggregate_output_tokens_per_second",
        "p50_first_chunk_ms",
        "p95_first_chunk_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p50_stream_tpot_ms",
        "p95_stream_tpot_ms",
    )
    return [
        {field: run.get(field) for field in row_fields}
        for run in sorted(scenario_runs, key=lambda run: run["run_order"])
    ]


def _vllm_server_concurrent_summary_rows(
    scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "prompt_tokens_min",
        "prompt_tokens_max",
        "prompt_tokens_mean",
        "total_prompt_tokens",
        "estimated_peak_sequence_tokens",
        "batch_wall_ms",
        "total_output_tokens",
        "aggregate_output_tokens_per_second",
        "p50_first_content_ms",
        "p95_first_content_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p50_stream_tpot_ms",
        "p95_stream_tpot_ms",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in scenarios
    ]


def _metric_distribution(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, float | int | None]:
    values = [
        float(record[field])
        for record in records
        if record.get(field) is not None
    ]
    if not values:
        return {
            f"{field}_count": 0,
            f"{field}_min": None,
            f"{field}_max": None,
            f"{field}_mean": None,
            f"{field}_median": None,
            f"{field}_p95": None,
            f"{field}_cv": None,
        }

    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    stddev = variance**0.5
    return {
        f"{field}_count": len(values),
        f"{field}_min": min(values),
        f"{field}_max": max(values),
        f"{field}_mean": mean_value,
        f"{field}_median": _percentile(values, 50),
        f"{field}_p95": _percentile(values, 95),
        f"{field}_cv": stddev / mean_value if mean_value else None,
    }


def _compare_vllm_sweep_csvs(
    cold_csv: Path,
    prefix_csv: Path,
    cold_label: str,
    prefix_label: str,
) -> dict[str, Any]:
    cold_rows = _read_csv_by_key(cold_csv, "scenario_id")
    prefix_rows = _read_csv_by_key(prefix_csv, "scenario_id")
    scenario_ids = sorted(set(cold_rows) & set(prefix_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for comparison")

    rows = []
    for scenario_id in scenario_ids:
        cold = cold_rows[scenario_id]
        prefix = prefix_rows[scenario_id]
        throughput_cold = _float_field(cold, "aggregate_output_tokens_per_second_median")
        throughput_prefix = _float_field(prefix, "aggregate_output_tokens_per_second_median")
        first_chunk_cold = _float_field(cold, "p95_first_chunk_ms_median")
        first_chunk_prefix = _float_field(prefix, "p95_first_chunk_ms_median")
        latency_cold = _float_field(cold, "p95_latency_ms_median")
        latency_prefix = _float_field(prefix, "p95_latency_ms_median")
        tpot_cold = _float_field(cold, "p95_stream_tpot_ms_median")
        tpot_prefix = _float_field(prefix, "p95_stream_tpot_ms_median")
        rows.append(
            {
                "scenario_id": scenario_id,
                "prompt_profile": cold["prompt_profile"],
                "request_count": int(cold["request_count"]),
                "max_new_tokens": int(cold["max_new_tokens"]),
                "prompt_tokens_mean": _float_field(cold, "prompt_tokens_mean"),
                "estimated_peak_sequence_tokens_median": _float_field(
                    cold,
                    "estimated_peak_sequence_tokens_median",
                ),
                "cold_output_tokens_per_second_median": throughput_cold,
                "prefix_output_tokens_per_second_median": throughput_prefix,
                "output_tokens_per_second_delta": _delta(throughput_prefix, throughput_cold),
                "output_tokens_per_second_ratio": _ratio(throughput_prefix, throughput_cold),
                "cold_p95_first_chunk_ms_median": first_chunk_cold,
                "prefix_p95_first_chunk_ms_median": first_chunk_prefix,
                "p95_first_chunk_ms_delta": _delta(first_chunk_prefix, first_chunk_cold),
                "p95_first_chunk_ms_ratio": _ratio(first_chunk_prefix, first_chunk_cold),
                "cold_p95_latency_ms_median": latency_cold,
                "prefix_p95_latency_ms_median": latency_prefix,
                "p95_latency_ms_delta": _delta(latency_prefix, latency_cold),
                "p95_latency_ms_ratio": _ratio(latency_prefix, latency_cold),
                "cold_p95_stream_tpot_ms_median": tpot_cold,
                "prefix_p95_stream_tpot_ms_median": tpot_prefix,
                "p95_stream_tpot_ms_delta": _delta(tpot_prefix, tpot_cold),
                "p95_stream_tpot_ms_ratio": _ratio(tpot_prefix, tpot_cold),
                "cold_throughput_cv": _float_field(
                    cold,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "prefix_throughput_cv": _float_field(
                    prefix,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "cold_latency_cv": _float_field(cold, "p95_latency_ms_cv"),
                "prefix_latency_cv": _float_field(prefix, "p95_latency_ms_cv"),
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-compare",
        "cold_label": cold_label,
        "prefix_label": prefix_label,
        "cold_csv": str(cold_csv),
        "prefix_csv": str(prefix_csv),
        "scenario_count": len(rows),
        "rows": rows,
    }


def _read_csv_by_key(path: Path, key: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {row[key]: row for row in reader}


def _float_field(row: dict[str, str], field: str) -> float | None:
    value = row.get(field)
    if value in {None, ""}:
        return None
    return float(value)


def _delta(new_value: float | None, baseline_value: float | None) -> float | None:
    if new_value is None or baseline_value is None:
        return None
    return new_value - baseline_value


def _ratio(new_value: float | None, baseline_value: float | None) -> float | None:
    if new_value is None or baseline_value in {None, 0.0}:
        return None
    return new_value / baseline_value


def _tail_lines(lines: list[str], count: int) -> list[str]:
    if count <= 0:
        return []
    return lines[-count:]


def _model_config_path(name: str, root: Path = ROOT) -> Path:
    filename = name if name.endswith(".json") else f"{name}.json"
    return root / "configs" / "models" / filename


def _capacity_config_path(name: str, root: Path = ROOT) -> Path:
    filename = name if name.endswith(".json") else f"{name}.json"
    return root / "configs" / "capacity" / filename


def _write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(records[0])
    extra_fields = sorted(set().union(*(record.keys() for record in records)) - set(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extra_fields])
        writer.writeheader()
        writer.writerows(records)


def _tokenize_prompt(tokenizer: Any, prompt: str) -> tuple[dict[str, Any], str]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ),
            "chat_template",
        )
    return tokenizer(prompt, return_tensors="pt"), "plain"


def _format_prompt_for_generation(tokenizer: Any, prompt: str) -> tuple[str, str]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return (
            tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            ),
            "chat_template",
        )
    return prompt, "plain"


def _manual_greedy_decode(
    model: Any,
    inputs: dict[str, Any],
    max_new_tokens: int,
    torch_module: Any,
) -> dict[str, Any]:
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch_module.ones_like(input_ids)

    generated_tokens = []
    with torch_module.inference_mode():
        _synchronize_if_cuda(torch_module)
        started = time.perf_counter()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        past_key_values = outputs.past_key_values
        _synchronize_if_cuda(torch_module)
        ttft_ms = (time.perf_counter() - started) * 1000

        decode_started = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            attention_mask = torch_module.cat(
                [
                    attention_mask,
                    torch_module.ones(
                        (attention_mask.shape[0], 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=-1,
            )
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            past_key_values = outputs.past_key_values
        _synchronize_if_cuda(torch_module)
        decode_ms = (time.perf_counter() - decode_started) * 1000

    return {
        "generated_ids": torch_module.cat(generated_tokens, dim=-1),
        "ttft_ms": ttft_ms,
        "decode_ms": decode_ms,
    }


def _synchronize_if_cuda(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def _serialize_vllm_metrics(output: Any) -> dict[str, float | None]:
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return {}
    metric_names = (
        "arrival_time",
        "first_scheduled_time",
        "first_token_time",
        "last_token_time",
        "finished_time",
        "scheduler_time",
        "model_forward_time",
        "model_execute_time",
    )
    result: dict[str, float | None] = {}
    for name in metric_names:
        value = getattr(metrics, name, None)
        result[name] = float(value) if isinstance(value, int | float) else None
    return result


def _duration_from_metrics_ms(
    metrics: dict[str, float | None],
    start_key: str,
    end_key: str,
) -> float | None:
    start = metrics.get(start_key)
    end = metrics.get(end_key)
    if start is None or end is None:
        return None
    return max(0.0, (end - start) * 1000)


def _percentile(values: list[float | None], percentile: int) -> float | None:
    clean_values = sorted(value for value in values if value is not None)
    if not clean_values:
        return None
    rank = max(1, int((percentile / 100) * len(clean_values) + 0.999999))
    return clean_values[min(rank - 1, len(clean_values) - 1)]


def _run_command(command: list[str]) -> str:
    import subprocess

    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as error:
        return error.output.strip()
