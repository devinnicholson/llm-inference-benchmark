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
DEFAULT_HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_INFERENCE_PROMPT = "Explain KV cache in LLM inference in two concise sentences."
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
    .uv_pip_install("vllm==0.21.0", "huggingface-hub==0.36.0")
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

    if mode != "sweep":
        raise ValueError(
            "mode must be 'sweep', 'gpu-probe', 'tiny-inference', "
            "'vllm-inference', or 'vllm-streaming'"
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


def _run_command(command: list[str]) -> str:
    import subprocess

    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as error:
        return error.output.strip()
