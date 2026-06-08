from __future__ import annotations

import csv
import json
import statistics
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
DEFAULT_VLLM_CACHE_METRICS_PROBE_OUTPUT = "results/modal-vllm-cache-metrics-probe"
DEFAULT_VLLM_CACHE_METRICS_SMOKE_OUTPUT = "results/modal-vllm-cache-metrics-smoke"
DEFAULT_VLLM_SWEEP_OUTPUT = "results/modal-vllm-sweep"
DEFAULT_VLLM_SERVER_OUTPUT = "results/modal-vllm-server-streaming"
DEFAULT_VLLM_SERVER_CONCURRENT_OUTPUT = "results/modal-vllm-server-concurrent"
DEFAULT_VLLM_SERVER_SWEEP_OUTPUT = "results/modal-vllm-server-sweep"
DEFAULT_VLLM_SERVER_SWEEP_COMPARE_OUTPUT = "results/modal-vllm-server-sweep-compare"
DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT = "results/modal-vllm-server-async-paired"
DEFAULT_VLLM_SERVER_ASYNC_PAIRED_SERVER_FIRST_OUTPUT = (
    "results/modal-vllm-server-async-paired-server-first"
)
DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_OUTPUT = (
    "results/modal-vllm-server-async-phase-order-compare"
)
DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_DIRS = (
    "results/modal-vllm-server-async-phase-order-compare,"
    "results/modal-vllm-server-async-phase-order-compare-trial2,"
    "results/modal-vllm-server-async-phase-order-compare-trial3"
)
DEFAULT_VLLM_SERVER_ASYNC_MULTITRIAL_OUTPUT = (
    "results/modal-vllm-server-async-multitrial-aggregate"
)
DEFAULT_VLLM_SERVER_ASYNC_LONG_PHASE_ORDER_COMPARE_OUTPUT = (
    "results/modal-vllm-server-async-phase-order-compare-long"
)
DEFAULT_VLLM_SERVER_ASYNC_WORKLOAD_COMPARE_OUTPUT = (
    "results/modal-vllm-server-async-workload-compare"
)
DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT = "results/modal-vllm-prefix-cache-sweep"
DEFAULT_VLLM_PREFIX_CACHE_COMPARE_OUTPUT = "results/modal-vllm-prefix-cache-compare"
DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT = "results/modal-vllm-prefix-cache-paired"
DEFAULT_VLLM_PREFIX_CACHE_PAIRED_CACHE_FIRST_OUTPUT = (
    "results/modal-vllm-prefix-cache-paired-cache-first"
)
DEFAULT_VLLM_PREFIX_CACHE_PHASE_ORDER_COMPARE_OUTPUT = (
    "results/modal-vllm-prefix-cache-phase-order-compare"
)
DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_OUTPUT = (
    "results/modal-vllm-prefix-cache-profile-control"
)
DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_DIRS = (
    "results/modal-vllm-prefix-cache-long-control-profile-control"
)
DEFAULT_VLLM_PREFIX_CACHE_PROFILE_MULTITRIAL_OUTPUT = (
    "results/modal-vllm-prefix-cache-long-control-multitrial"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_METRICS_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-metrics"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WARM_WINDOW_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-warm-window"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_NEUTRAL_WARMUP_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-neutral-warmup"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_STABILITY_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-stability-summary"
)
DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WINDOW_OUTPUT = (
    "results/modal-vllm-prefix-cache-isolated-window-summary"
)
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
VALID_VLLM_PROMPT_PROFILES = {
    "short",
    "long",
    "mixed",
    "shared_prefix",
    "shared_prefix_long",
    "matched_unique_prefix",
    "neutral_long",
}
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
    timeout=1800,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_prefix_cache_paired_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    warmup_prompt_profile: str = "",
    phase_order: str = "cold_first",
    collect_cache_metrics: bool = False,
    kv_cache_metrics_sample: float = 1.0,
) -> dict[str, Any]:
    import asyncio
    import contextlib
    import gc
    import inspect
    import io
    import logging
    import platform
    import random
    import re
    import time

    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if kv_cache_metrics_sample <= 0:
        raise ValueError("kv_cache_metrics_sample must be positive")
    phase_order = phase_order.lower().replace("-", "_")
    if phase_order not in {"cold_first", "cache_first"}:
        raise ValueError("phase_order must be cold_first or cache_first")
    _validate_vllm_prompt_profiles(prompt_profile_values)
    warmup_prompt_profile_value = warmup_prompt_profile.strip().lower().replace("-", "_")
    if warmup_prompt_profile_value:
        _validate_vllm_prompt_profiles(
            [warmup_prompt_profile_value],
            label="warmup_prompt_profile",
        )

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

    prompt_format_started = time.perf_counter()

    def build_scenario_specs(
        profiles: list[str],
        scenario_id_prefix: str = "",
    ) -> list[dict[str, Any]]:
        specs = []
        for prompt_profile in profiles:
            for max_new_tokens in output_token_values:
                for request_count in request_count_values:
                    prompt_records = []
                    prompts = _select_sweep_prompts(request_count, prompt_profile)
                    for index, prompt in enumerate(prompts):
                        formatted_prompt, prompt_format = _format_prompt_for_generation(
                            tokenizer,
                            prompt,
                        )
                        prompt_records.append(
                            {
                                "request_id": (
                                    f"{prompt_profile}-out{max_new_tokens}-"
                                    f"n{request_count}-{index:02d}"
                                ),
                                "prompt": prompt,
                                "formatted_prompt": formatted_prompt,
                                "prompt_format": prompt_format,
                                "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
                            }
                        )
                    prompt_tokens = [
                        record["prompt_tokens"] for record in prompt_records
                    ]
                    scenario_id = (
                        f"{prompt_profile}_out{max_new_tokens}_n{request_count}"
                    )
                    if scenario_id_prefix:
                        scenario_id = f"{scenario_id_prefix}_{scenario_id}"
                    specs.append(
                        {
                            "scenario_id": scenario_id,
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
        return specs

    scenario_specs = build_scenario_specs(prompt_profile_values)
    warmup_scenario_specs = (
        build_scenario_specs(
            [warmup_prompt_profile_value],
            scenario_id_prefix="warmup",
        )
        if warmup_prompt_profile_value
        else scenario_specs
    )
    prompt_format_ms = (time.perf_counter() - prompt_format_started) * 1000

    max_request_count = max(request_count_values)
    max_output_tokens = max(output_token_values)
    all_shape_specs = scenario_specs + warmup_scenario_specs
    max_prompt_tokens = max(spec["prompt_tokens_max"] for spec in all_shape_specs)
    max_model_len = max(1024, max_prompt_tokens + max_output_tokens + 32)
    max_num_batched_tokens = max(2048, max_request_count * max_model_len)
    scenario_plan = []
    for repeat_index in range(repeats):
        for spec in scenario_specs:
            scenario_plan.append((repeat_index, spec))
    random.Random(scenario_seed).shuffle(scenario_plan)
    plan_summary = [
        {
            "run_order": run_order,
            "repeat_index": repeat_index,
            "scenario_id": spec["scenario_id"],
        }
        for run_order, (repeat_index, spec) in enumerate(scenario_plan)
    ]

    async def call_engine_log_stats(engine: Any, label: str) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        captured_logs: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_logs.append(self.format(record))

        handler = CaptureHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        capture_loggers = [
            logging.getLogger(),
            logging.getLogger("vllm"),
            logging.getLogger("vllm.v1.metrics.loggers"),
        ]
        started = time.perf_counter()
        try:
            for logger in capture_loggers:
                logger.addHandler(handler)
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                result = engine.do_log_stats()
                if inspect.isawaitable(result):
                    result = await result
            return {
                "label": label,
                "ok": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(result).__name__,
                "return_repr": repr(result)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        except Exception as error:  # pragma: no cover - remote metrics probe
            return {
                "label": label,
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(error).__name__,
                "return_repr": str(error)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        finally:
            for logger in capture_loggers:
                logger.removeHandler(handler)

    def parse_engine_log_metrics(log_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        seen_lines = set()
        patterns = {
            "avg_prompt_throughput_tokens_per_s": r"Avg prompt throughput: ([0-9.]+) tokens/s",
            "avg_generation_throughput_tokens_per_s": (
                r"Avg generation throughput: ([0-9.]+) tokens/s"
            ),
            "gpu_kv_cache_usage_pct": r"GPU KV cache usage: ([0-9.]+)%",
            "prefix_cache_hit_rate_pct": r"Prefix cache hit rate: ([0-9.]+)%",
        }
        for call in log_stats:
            for line in call.get("captured_logs", []):
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                row: dict[str, Any] = {
                    "label": call["label"],
                    "line": line,
                }
                matched = False
                for field, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        row[field] = float(match.group(1))
                        matched = True
                if matched:
                    rows.append(row)
        return rows

    def latest_cache_metric_value(
        metrics: list[dict[str, Any]],
        field: str,
    ) -> float | None:
        for row in reversed(metrics):
            value = row.get(field)
            if value is not None:
                return float(value)
        return None

    async def run_engine_phase(
        *,
        enable_prefix_caching: bool,
        phase_label: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        metrics_args = (
            {
                "kv_cache_metrics": True,
                "kv_cache_metrics_sample": kv_cache_metrics_sample,
                "disable_log_stats": False,
            }
            if collect_cache_metrics
            else {}
        )
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
            **metrics_args,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000

        async def stream_one(
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
                request_id=f"{phase_label}-{run_id}-{record['request_id']}",
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
                "generated_text": "".join(generated_text_parts),
            }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
            max_tokens_override: int | None = None,
        ) -> dict[str, Any]:
            max_tokens = max_tokens_override or spec["max_new_tokens"]
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(
                    stream_one(
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
            cache_log_stats = None
            cache_metrics: list[dict[str, Any]] = []
            if collect_cache_metrics:
                cache_log_stats = await call_engine_log_stats(
                    engine,
                    f"{phase_label}_{run_id}_{spec['scenario_id']}",
                )
                cache_metrics = parse_engine_log_metrics([cache_log_stats])
            return {
                "run_id": run_id,
                "repeat_index": repeat_index,
                "run_order": run_order,
                "scenario_id": spec["scenario_id"],
                "prompt_profile": spec["prompt_profile"],
                "request_count": spec["request_count"],
                "max_new_tokens": max_tokens,
                "prompt_format": spec["prompt_format"],
                "prompt_tokens_min": spec["prompt_tokens_min"],
                "prompt_tokens_max": spec["prompt_tokens_max"],
                "prompt_tokens_mean": spec["prompt_tokens_mean"],
                "total_prompt_tokens": spec["total_prompt_tokens"],
                "estimated_peak_sequence_tokens": sum(
                    request["total_sequence_tokens"] for request in request_results
                ),
                **summary,
                "cache_metrics_log_stats": cache_log_stats,
                "cache_metrics": cache_metrics,
                "prefix_cache_hit_rate_pct": latest_cache_metric_value(
                    cache_metrics,
                    "prefix_cache_hit_rate_pct",
                ),
                "gpu_kv_cache_usage_pct": latest_cache_metric_value(
                    cache_metrics,
                    "gpu_kv_cache_usage_pct",
                ),
                "avg_prompt_throughput_tokens_per_s": latest_cache_metric_value(
                    cache_metrics,
                    "avg_prompt_throughput_tokens_per_s",
                ),
                "avg_generation_throughput_tokens_per_s": latest_cache_metric_value(
                    cache_metrics,
                    "avg_generation_throughput_tokens_per_s",
                ),
                "requests": request_results,
            }

        async def run_warmups() -> dict[str, Any]:
            started = time.perf_counter()
            warmup_summaries = []
            run_order = 0
            for warmup_index in range(warmup_runs):
                for spec in warmup_scenario_specs:
                    result = await run_scenario(
                        spec,
                        repeat_index=-(warmup_index + 1),
                        run_order=run_order,
                        max_tokens_override=1,
                    )
                    warmup_summaries.append(
                        {
                            "warmup_index": warmup_index,
                            "scenario_id": spec["scenario_id"],
                            "prompt_profile": spec["prompt_profile"],
                            "request_count": spec["request_count"],
                            "max_new_tokens": 1,
                            "prompt_tokens_mean": spec["prompt_tokens_mean"],
                            "total_prompt_tokens": spec["total_prompt_tokens"],
                            "batch_wall_ms": result["batch_wall_ms"],
                            "total_output_tokens": result["total_output_tokens"],
                        }
                    )
                    run_order += 1
            return {
                "warmup_runs": warmup_runs,
                "warmup_scenario_runs": len(warmup_summaries),
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
                "summaries": warmup_summaries,
            }

        try:
            warmup_result = await run_warmups()
            warmup_cache_log_stats = None
            warmup_cache_metrics: list[dict[str, Any]] = []
            if collect_cache_metrics:
                warmup_cache_log_stats = await call_engine_log_stats(
                    engine,
                    f"{phase_label}_after_warmup",
                )
                warmup_cache_metrics = parse_engine_log_metrics([warmup_cache_log_stats])
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
            "phase_label": phase_label,
            "enable_prefix_caching": enable_prefix_caching,
            "engine_load_ms": engine_load_ms,
            "warmup": warmup_result,
            "warmup_cache_metrics_log_stats": warmup_cache_log_stats,
            "warmup_cache_metrics": warmup_cache_metrics,
            "scenario_runs": scenario_runs,
        }

    def release_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2.0)

    if phase_order == "cold_first":
        cold_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=False, phase_label="cold")
        )
        release_cuda_memory()
        cache_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=True, phase_label="cache")
        )
    else:
        cache_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=True, phase_label="cache")
        )
        release_cuda_memory()
        cold_result = asyncio.run(
            run_engine_phase(enable_prefix_caching=False, phase_label="cold")
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

    paired_runs = _make_vllm_prefix_cache_paired_rows(
        cold_result["scenario_runs"],
        cache_result["scenario_runs"],
    )
    paired_scenarios = _aggregate_vllm_prefix_cache_paired_scenarios(paired_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-prefix-cache-paired",
        "backend": "vllm-paired-prefix-cache",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "warmup_runs": int(warmup_runs),
        "warmup_prompt_profile": warmup_prompt_profile_value or None,
        "warmup_scenario_count": len(warmup_scenario_specs) if warmup_runs else 0,
        "phase_order": phase_order,
        "collect_cache_metrics": bool(collect_cache_metrics),
        "kv_cache_metrics_sample": float(kv_cache_metrics_sample),
        "scenario_count": len(paired_scenarios),
        "paired_run_count": len(paired_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "cold_engine_load_ms": cold_result["engine_load_ms"],
        "cache_engine_load_ms": cache_result["engine_load_ms"],
        "cold_warmup": cold_result["warmup"],
        "cache_warmup": cache_result["warmup"],
        "cold_warmup_cache_metrics": cold_result["warmup_cache_metrics"],
        "cache_warmup_cache_metrics": cache_result["warmup_cache_metrics"],
        "cold_warmup_cache_metrics_log_stats": cold_result[
            "warmup_cache_metrics_log_stats"
        ],
        "cache_warmup_cache_metrics_log_stats": cache_result[
            "warmup_cache_metrics_log_stats"
        ],
        "scenario_plan": plan_summary,
        "summary": _vllm_prefix_cache_paired_summary_rows(paired_scenarios),
        "paired_runs": paired_runs,
        "paired_scenarios": paired_scenarios,
        "cold_scenario_runs": cold_result["scenario_runs"],
        "cache_scenario_runs": cache_result["scenario_runs"],
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "mean_cache_to_cold_throughput_ratio": _mean_present(
            row["cache_to_cold_output_tokens_per_second_ratio"] for row in paired_runs
        ),
        "mean_cache_to_cold_first_event_ratio": _mean_present(
            row["cache_to_cold_p95_first_event_ms_ratio"] for row in paired_runs
        ),
        "mean_cache_to_cold_latency_ratio": _mean_present(
            row["cache_to_cold_p95_latency_ms_ratio"] for row in paired_runs
        ),
        "mean_cache_to_cold_tpot_ratio": _mean_present(
            row["cache_to_cold_p95_stream_tpot_ms_ratio"] for row in paired_runs
        ),
        "note": (
            "One Modal worker runs two in-process AsyncLLM engines with prefix "
            f"caching off and on in phase order {phase_order}. Paired rows "
            "compare matching scenario_id and repeat_index."
        ),
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
    timeout=600,
)
def run_vllm_cache_metrics_probe_remote(
    hf_model: str = DEFAULT_HF_MODEL,
) -> dict[str, Any]:
    import inspect
    import platform

    import vllm
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    def signature_parameters(target: Any) -> list[str]:
        try:
            return list(inspect.signature(target).parameters)
        except (TypeError, ValueError):
            return []

    def dataclass_fields(target: Any) -> list[str]:
        fields = getattr(target, "__dataclass_fields__", {})
        return sorted(fields) if fields else []

    def relevant_names(names: list[str]) -> list[str]:
        markers = ("cache", "metric", "observ", "stat", "log")
        return sorted(
            name for name in names
            if any(marker in name.lower() for marker in markers)
        )

    async_engine_args_parameters = signature_parameters(AsyncEngineArgs)
    async_engine_args_fields = dataclass_fields(AsyncEngineArgs)
    async_llm_relevant_methods = relevant_names(
        [name for name in dir(AsyncLLM) if not name.startswith("_")]
    )
    observability_parameters: list[str] = []
    observability_fields: list[str] = []
    observability_error = ""
    try:
        from vllm.config import ObservabilityConfig

        observability_parameters = signature_parameters(ObservabilityConfig)
        observability_fields = dataclass_fields(ObservabilityConfig)
    except Exception as error:  # pragma: no cover - remote environment probe
        observability_error = f"{type(error).__name__}: {error}"

    constructor_checks = []
    for kwargs in (
        {"kv_cache_metrics": True},
        {"kv_cache_metrics": True, "kv_cache_metrics_sample": 1.0},
        {"disable_log_stats": False},
    ):
        try:
            AsyncEngineArgs(model=hf_model, **kwargs)
            constructor_checks.append({"kwargs": kwargs, "accepted": True, "error": ""})
        except Exception as error:  # pragma: no cover - remote environment probe
            constructor_checks.append(
                {
                    "kwargs": kwargs,
                    "accepted": False,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-cache-metrics-probe",
        "model_id": hf_model,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "async_engine_args_parameters": async_engine_args_parameters,
        "async_engine_args_fields": async_engine_args_fields,
        "async_engine_args_relevant_parameters": relevant_names(
            async_engine_args_parameters
        ),
        "async_engine_args_relevant_fields": relevant_names(async_engine_args_fields),
        "async_llm_relevant_methods": async_llm_relevant_methods,
        "observability_config_parameters": observability_parameters,
        "observability_config_fields": observability_fields,
        "observability_config_relevant_parameters": relevant_names(
            observability_parameters
        ),
        "observability_config_relevant_fields": relevant_names(observability_fields),
        "observability_config_error": observability_error,
        "constructor_checks": constructor_checks,
        "note": (
            "Remote introspection of vLLM constructor surfaces related to cache "
            "metrics, observability, and logging. This does not run inference."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=1200,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_cache_metrics_smoke_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    prompt_profile: str = "shared_prefix_long",
    request_count: int = 4,
    max_new_tokens: int = 8,
) -> dict[str, Any]:
    import asyncio
    import contextlib
    import inspect
    import io
    import logging
    import platform
    import re
    import time

    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM
    import vllm

    if request_count <= 0:
        raise ValueError("request_count must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    prompt_profile = prompt_profile.lower().replace("-", "_")
    _validate_vllm_prompt_profiles([prompt_profile], label="prompt_profile")

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    prompts = _select_sweep_prompts(request_count, prompt_profile)
    prompt_records = []
    for index, prompt in enumerate(prompts):
        formatted_prompt, prompt_format = _format_prompt_for_generation(tokenizer, prompt)
        prompt_records.append(
            {
                "request_id": f"{prompt_profile}-metrics-n{request_count}-{index:02d}",
                "prompt": prompt,
                "formatted_prompt": formatted_prompt,
                "prompt_format": prompt_format,
                "prompt_tokens": len(tokenizer.encode(formatted_prompt)),
            }
        )
    max_prompt_tokens = max(record["prompt_tokens"] for record in prompt_records)
    max_model_len = max(1024, max_prompt_tokens + max_new_tokens + 32)
    max_num_batched_tokens = max(2048, request_count * max_model_len)

    def relevant_object_attrs(target: Any) -> list[dict[str, Any]]:
        rows = []
        markers = ("cache", "metric", "stat", "log")
        for name in dir(target):
            if name.startswith("__") or not any(marker in name.lower() for marker in markers):
                continue
            try:
                value = getattr(target, name)
                value_type = type(value).__name__
                if callable(value):
                    value_repr = f"<callable {value_type}>"
                else:
                    value_repr = repr(value)
            except Exception as error:  # pragma: no cover - remote object probe
                value_type = type(error).__name__
                value_repr = f"<error {error}>"
            rows.append(
                {
                    "name": name,
                    "type": value_type,
                    "repr": value_repr[:500],
                }
            )
        return rows

    async def call_log_stats(engine: Any, label: str) -> dict[str, Any]:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        captured_logs: list[str] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                captured_logs.append(self.format(record))

        handler = CaptureHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        capture_loggers = [
            logging.getLogger(),
            logging.getLogger("vllm"),
            logging.getLogger("vllm.v1.metrics.loggers"),
        ]
        started = time.perf_counter()
        try:
            for logger in capture_loggers:
                logger.addHandler(handler)
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                result = engine.do_log_stats()
                if inspect.isawaitable(result):
                    result = await result
            return {
                "label": label,
                "ok": True,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(result).__name__,
                "return_repr": repr(result)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        except Exception as error:  # pragma: no cover - remote object probe
            return {
                "label": label,
                "ok": False,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "return_type": type(error).__name__,
                "return_repr": str(error)[:1000],
                "stdout": stdout_buffer.getvalue()[-4000:],
                "stderr": stderr_buffer.getvalue()[-4000:],
                "captured_logs": captured_logs[-20:],
            }
        finally:
            for logger in capture_loggers:
                logger.removeHandler(handler)

    def parse_log_metrics(log_stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = []
        patterns = {
            "avg_prompt_throughput_tokens_per_s": r"Avg prompt throughput: ([0-9.]+) tokens/s",
            "avg_generation_throughput_tokens_per_s": (
                r"Avg generation throughput: ([0-9.]+) tokens/s"
            ),
            "gpu_kv_cache_usage_pct": r"GPU KV cache usage: ([0-9.]+)%",
            "prefix_cache_hit_rate_pct": r"Prefix cache hit rate: ([0-9.]+)%",
        }
        for call in log_stats:
            for line in call.get("captured_logs", []):
                row: dict[str, Any] = {
                    "label": call["label"],
                    "line": line,
                }
                matched = False
                for field, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        row[field] = float(match.group(1))
                        matched = True
                if matched:
                    rows.append(row)
        return rows

    async def run_smoke() -> dict[str, Any]:
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=request_count,
            gpu_memory_utilization=0.50,
            enable_prefix_caching=True,
            kv_cache_metrics=True,
            kv_cache_metrics_sample=1.0,
            disable_log_stats=False,
            enforce_eager=True,
            trust_remote_code=False,
        )
        started = time.perf_counter()
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000
        log_stats = []
        try:
            log_stats.append(await call_log_stats(engine, "after_engine_load"))
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )

            async def stream_one(record: dict[str, Any]) -> dict[str, Any]:
                request_started = time.perf_counter()
                first_chunk_ms: float | None = None
                finished_ms: float | None = None
                generated_tokens = 0
                chunk_count = 0
                async for output in engine.generate(
                    request_id=f"metrics-smoke-{record['request_id']}",
                    prompt=record["formatted_prompt"],
                    sampling_params=sampling_params,
                ):
                    elapsed_ms = (time.perf_counter() - request_started) * 1000
                    chunk_count += 1
                    for completion in output.outputs:
                        token_ids = list(completion.token_ids or [])
                        if first_chunk_ms is None and (completion.text or token_ids):
                            first_chunk_ms = elapsed_ms
                        generated_tokens += len(token_ids)
                    if output.finished:
                        finished_ms = elapsed_ms
                        break
                stream_wall_ms = finished_ms or (
                    (time.perf_counter() - request_started) * 1000
                )
                decode_tokens_after_first = max(generated_tokens - 1, 0)
                decode_after_first_ms = (
                    stream_wall_ms - first_chunk_ms
                    if first_chunk_ms is not None
                    else None
                )
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
                }

            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(stream_one(record) for record in prompt_records)
            )
            batch_wall_ms = (time.perf_counter() - batch_started) * 1000
            log_stats.append(await call_log_stats(engine, "after_shared_prefix_batch"))
            parsed_log_metrics = parse_log_metrics(log_stats)
            engine_attrs = relevant_object_attrs(engine)
            engine_core_attrs = (
                relevant_object_attrs(engine.engine_core)
                if hasattr(engine, "engine_core")
                else []
            )
        finally:
            engine.shutdown()

        return {
            "engine_load_ms": engine_load_ms,
            "request_results": request_results,
            "batch_wall_ms": batch_wall_ms,
            "summary": _summarize_stream_requests(request_results, batch_wall_ms),
            "log_stats_calls": log_stats,
            "parsed_log_metrics": parsed_log_metrics,
            "engine_relevant_attrs": engine_attrs,
            "engine_core_relevant_attrs": engine_core_attrs,
        }

    nvidia_smi_before = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    smoke = asyncio.run(run_smoke())
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
        "mode": "vllm-cache-metrics-smoke",
        "backend": "vllm-async-engine",
        "model_id": hf_model,
        "prompt_profile": prompt_profile,
        "request_count": request_count,
        "max_new_tokens": max_new_tokens,
        "prompt_tokens_min": min(record["prompt_tokens"] for record in prompt_records),
        "prompt_tokens_max": max_prompt_tokens,
        "prompt_tokens_mean": (
            sum(record["prompt_tokens"] for record in prompt_records) / len(prompt_records)
        ),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "enable_prefix_caching": True,
        "kv_cache_metrics": True,
        "kv_cache_metrics_sample": 1.0,
        "disable_log_stats": False,
        "smoke": smoke,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "note": (
            "Tiny GPU-backed metrics smoke. The run enables vLLM KV-cache "
            "metrics, executes one shared-prefix batch, calls do_log_stats "
            "before and after the batch, and records visible cache/stat/log "
            "attributes on AsyncLLM and engine_core."
        ),
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
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    _validate_vllm_prompt_profiles(prompt_profile_values)

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
    prompt_profile = prompt_profile.lower().replace("-", "_")
    _validate_vllm_prompt_profiles([prompt_profile], label="prompt_profile")
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


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=2400,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_sweep_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    ready_timeout_s: int = 600,
) -> dict[str, Any]:
    import asyncio
    import json
    import platform
    import random
    import subprocess
    import threading
    import time

    import httpx
    from transformers import AutoTokenizer

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")
    _validate_vllm_prompt_profiles(prompt_profile_values)

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
                for index, prompt in enumerate(_select_sweep_prompts(request_count, prompt_profile)):
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
        client: httpx.AsyncClient,
        scenario_id: str,
        record: dict[str, Any],
        max_tokens: int,
        batch_started: float,
    ) -> dict[str, Any]:
        request_body = {
            "model": hf_model,
            "messages": [{"role": "user", "content": record["prompt"]}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        chunks: list[dict[str, Any]] = []
        generated_text_parts: list[str] = []
        usage: dict[str, Any] | None = None
        response_status_code: int | None = None
        first_content_ms: float | None = None
        request_started = time.perf_counter()
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
                now = time.perf_counter()
                elapsed_ms = (now - request_started) * 1000
                batch_elapsed_ms = (now - batch_started) * 1000
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
            "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "usage": usage,
        }

    async def run_scenario(
        spec: dict[str, Any],
        repeat_index: int,
        run_order: int,
        max_tokens_override: int | None = None,
    ) -> dict[str, Any]:
        max_tokens = max_tokens_override or spec["max_new_tokens"]
        run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
        batch_started = time.perf_counter()
        async with httpx.AsyncClient(timeout=None) as client:
            request_results = await asyncio.gather(
                *(
                    stream_one(
                        client,
                        spec["scenario_id"],
                        record,
                        max_tokens,
                        batch_started,
                    )
                    for record in spec["prompt_records"]
                )
            )
        batch_wall_ms = (time.perf_counter() - batch_started) * 1000
        summary = _summarize_server_stream_requests(request_results, batch_wall_ms)
        return {
            "run_id": run_id,
            "repeat_index": repeat_index,
            "run_order": run_order,
            "scenario_id": spec["scenario_id"],
            "prompt_profile": spec["prompt_profile"],
            "request_count": spec["request_count"],
            "max_new_tokens": max_tokens,
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

    async def run_warmups() -> dict[str, Any]:
        started = time.perf_counter()
        warmup_summaries = []
        run_order = 0
        for warmup_index in range(warmup_runs):
            for spec in scenario_specs:
                result = await run_scenario(
                    spec,
                    repeat_index=-(warmup_index + 1),
                    run_order=run_order,
                    max_tokens_override=1,
                )
                warmup_summaries.append(
                    {
                        "warmup_index": warmup_index,
                        "scenario_id": spec["scenario_id"],
                        "request_count": spec["request_count"],
                        "batch_wall_ms": result["batch_wall_ms"],
                        "total_output_tokens": result["total_output_tokens"],
                    }
                )
                run_order += 1
        return {
            "warmup_runs": warmup_runs,
            "warmup_scenario_runs": len(warmup_summaries),
            "warmup_wall_ms": (time.perf_counter() - started) * 1000,
            "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
            "summaries": warmup_summaries,
        }

    ready_ms: float | None = None
    warmup_result: dict[str, Any] = {}
    scenario_runs: list[dict[str, Any]] = []
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

        warmup_result = asyncio.run(run_warmups())
        scenario_plan = []
        for repeat_index in range(repeats):
            for spec in scenario_specs:
                scenario_plan.append((repeat_index, spec))
        random.Random(scenario_seed).shuffle(scenario_plan)
        for run_order, (repeat_index, spec) in enumerate(scenario_plan):
            scenario_runs.append(
                asyncio.run(
                    run_scenario(
                        spec,
                        repeat_index=repeat_index,
                        run_order=run_order,
                    )
                )
            )
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

    scenarios = _aggregate_vllm_server_scenarios(scenario_specs, scenario_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-sweep",
        "backend": "vllm-openai-server",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "warmup_runs": int(warmup_runs),
        "scenario_count": len(scenarios),
        "scenario_run_count": len(scenario_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "server_ready_ms": ready_ms,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "warmup": warmup_result,
        "summary": _vllm_server_sweep_summary_rows(scenarios),
        "run_summary": _vllm_server_sweep_run_rows(scenario_runs),
        "scenarios": scenarios,
        "scenario_runs": scenario_runs,
        "server_command": command,
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_logs[:80],
        "server_logs_tail": _tail_lines(server_logs, 80),
        "note": (
            "One OpenAI-compatible vLLM server runs warmup/discard requests, "
            "then a seeded, repeated, shuffled concurrency sweep over "
            "request count, prompt profile, and output token budget."
        ),
    }


@app.function(
    image=vllm_image,
    gpu="T4",
    timeout=3600,
    volumes={HF_CACHE_PATH: hf_cache_volume, VLLM_CACHE_PATH: vllm_cache_volume},
)
def run_vllm_server_async_paired_remote(
    hf_model: str = DEFAULT_HF_MODEL,
    request_counts: str = DEFAULT_VLLM_SWEEP_REQUEST_COUNTS,
    prompt_profiles: str = DEFAULT_VLLM_SWEEP_PROMPT_PROFILES,
    output_tokens: str = DEFAULT_VLLM_SWEEP_OUTPUT_TOKENS,
    repeats: int = DEFAULT_VLLM_SWEEP_REPEATS,
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    warmup_runs: int = 1,
    phase_order: str = "async_first",
    ready_timeout_s: int = 600,
) -> dict[str, Any]:
    import asyncio
    import gc
    import json
    import platform
    import random
    import subprocess
    import threading
    import time

    import httpx
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    from vllm.v1.engine.async_llm import AsyncLLM

    request_count_values = _split_positive_int_csv(request_counts, "request_counts")
    prompt_profile_values = [
        profile.lower().replace("-", "_")
        for profile in _split_csv(prompt_profiles)
    ]
    output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if ready_timeout_s <= 0:
        raise ValueError("ready_timeout_s must be positive")
    phase_order = phase_order.lower().replace("-", "_")
    if phase_order not in {"async_first", "server_first"}:
        raise ValueError("phase_order must be async_first or server_first")
    _validate_vllm_prompt_profiles(prompt_profile_values)

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
                for index, prompt in enumerate(_select_sweep_prompts(request_count, prompt_profile)):
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
    scenario_plan = []
    for repeat_index in range(repeats):
        for spec in scenario_specs:
            scenario_plan.append((repeat_index, spec))
    random.Random(scenario_seed).shuffle(scenario_plan)
    plan_summary = [
        {
            "run_order": run_order,
            "repeat_index": repeat_index,
            "scenario_id": spec["scenario_id"],
        }
        for run_order, (repeat_index, spec) in enumerate(scenario_plan)
    ]

    async def run_async_phase() -> dict[str, Any]:
        started = time.perf_counter()
        engine_args = AsyncEngineArgs(
            model=hf_model,
            dtype="half",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_request_count,
            gpu_memory_utilization=0.50,
            enable_prefix_caching=False,
            enforce_eager=True,
            trust_remote_code=False,
        )
        engine = AsyncLLM.from_engine_args(engine_args)
        engine_load_ms = (time.perf_counter() - started) * 1000

        async def stream_one(
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
                request_id=f"paired-async-{run_id}-{record['request_id']}",
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
                "generated_text": "".join(generated_text_parts),
            }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
            max_tokens_override: int | None = None,
        ) -> dict[str, Any]:
            max_tokens = max_tokens_override or spec["max_new_tokens"]
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=0.0,
                output_kind=RequestOutputKind.DELTA,
            )
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            batch_started = time.perf_counter()
            request_results = await asyncio.gather(
                *(
                    stream_one(
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
                "max_new_tokens": max_tokens,
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

        async def run_warmups() -> dict[str, Any]:
            started = time.perf_counter()
            warmup_summaries = []
            run_order = 0
            for warmup_index in range(warmup_runs):
                for spec in scenario_specs:
                    result = await run_scenario(
                        spec,
                        repeat_index=-(warmup_index + 1),
                        run_order=run_order,
                        max_tokens_override=1,
                    )
                    warmup_summaries.append(
                        {
                            "warmup_index": warmup_index,
                            "scenario_id": spec["scenario_id"],
                            "request_count": spec["request_count"],
                            "batch_wall_ms": result["batch_wall_ms"],
                            "total_output_tokens": result["total_output_tokens"],
                        }
                    )
                    run_order += 1
            return {
                "warmup_runs": warmup_runs,
                "warmup_scenario_runs": len(warmup_summaries),
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
                "summaries": warmup_summaries,
            }

        try:
            warmup_result = await run_warmups()
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

    def run_server_phase() -> dict[str, Any]:
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
            client: httpx.AsyncClient,
            record: dict[str, Any],
            max_tokens: int,
            batch_started: float,
        ) -> dict[str, Any]:
            request_body = {
                "model": hf_model,
                "messages": [{"role": "user", "content": record["prompt"]}],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            generated_text_parts: list[str] = []
            usage: dict[str, Any] | None = None
            response_status_code: int | None = None
            first_content_ms: float | None = None
            chunk_count = 0
            request_started = time.perf_counter()
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
                    if event.get("usage") is not None:
                        usage = event["usage"]
                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if content and first_content_ms is None:
                            first_content_ms = elapsed_ms
                        if content:
                            generated_text_parts.append(content)
                        chunk_count += 1
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
                "batch_finished_ms": (time.perf_counter() - batch_started) * 1000,
                "chunk_count": chunk_count,
                "usage": usage,
            }

        async def run_scenario(
            spec: dict[str, Any],
            repeat_index: int,
            run_order: int,
            max_tokens_override: int | None = None,
        ) -> dict[str, Any]:
            max_tokens = max_tokens_override or spec["max_new_tokens"]
            run_id = f"rep{repeat_index:02d}-run{run_order:03d}"
            batch_started = time.perf_counter()
            async with httpx.AsyncClient(timeout=None) as client:
                request_results = await asyncio.gather(
                    *(
                        stream_one(
                            client,
                            record,
                            max_tokens,
                            batch_started,
                        )
                        for record in spec["prompt_records"]
                    )
                )
            batch_wall_ms = (time.perf_counter() - batch_started) * 1000
            summary = _summarize_server_stream_requests(request_results, batch_wall_ms)
            return {
                "run_id": run_id,
                "repeat_index": repeat_index,
                "run_order": run_order,
                "scenario_id": spec["scenario_id"],
                "prompt_profile": spec["prompt_profile"],
                "request_count": spec["request_count"],
                "max_new_tokens": max_tokens,
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

        async def run_warmups() -> dict[str, Any]:
            started = time.perf_counter()
            warmup_summaries = []
            run_order = 0
            for warmup_index in range(warmup_runs):
                for spec in scenario_specs:
                    result = await run_scenario(
                        spec,
                        repeat_index=-(warmup_index + 1),
                        run_order=run_order,
                        max_tokens_override=1,
                    )
                    warmup_summaries.append(
                        {
                            "warmup_index": warmup_index,
                            "scenario_id": spec["scenario_id"],
                            "request_count": spec["request_count"],
                            "batch_wall_ms": result["batch_wall_ms"],
                            "total_output_tokens": result["total_output_tokens"],
                        }
                    )
                    run_order += 1
            return {
                "warmup_runs": warmup_runs,
                "warmup_scenario_runs": len(warmup_summaries),
                "warmup_wall_ms": (time.perf_counter() - started) * 1000,
                "total_output_tokens": sum(row["total_output_tokens"] for row in warmup_summaries),
                "summaries": warmup_summaries,
            }

        ready_ms: float | None = None
        warmup_result: dict[str, Any] = {}
        scenario_runs: list[dict[str, Any]] = []
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

            warmup_result = asyncio.run(run_warmups())
            for run_order, (repeat_index, spec) in enumerate(scenario_plan):
                scenario_runs.append(
                    asyncio.run(
                        run_scenario(
                            spec,
                            repeat_index=repeat_index,
                            run_order=run_order,
                        )
                    )
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=20)
            log_thread.join(timeout=5)

        return {
            "server_ready_ms": ready_ms,
            "warmup": warmup_result,
            "scenario_runs": scenario_runs,
            "server_command": command,
            "server_logs_head": server_logs[:80],
            "server_logs_tail": _tail_lines(server_logs, 80),
        }

    def release_cuda_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        time.sleep(2.0)

    if phase_order == "async_first":
        async_result = asyncio.run(run_async_phase())
        release_cuda_memory()
        server_result = run_server_phase()
    else:
        server_result = run_server_phase()
        release_cuda_memory()
        async_result = asyncio.run(run_async_phase())

    hf_cache_volume.commit()
    vllm_cache_volume.commit()
    nvidia_smi_after = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )

    paired_runs = _make_vllm_server_async_paired_rows(
        async_result["scenario_runs"],
        server_result["scenario_runs"],
    )
    paired_scenarios = _aggregate_vllm_server_async_paired_scenarios(paired_runs)
    return {
        "schema_version": 1,
        "execution": "modal",
        "mode": "vllm-server-async-paired",
        "backend": "vllm-paired-async-and-openai-server",
        "model_id": hf_model,
        "request_counts": request_count_values,
        "prompt_profiles": prompt_profile_values,
        "output_tokens": output_token_values,
        "repeats": int(repeats),
        "scenario_seed": int(scenario_seed),
        "warmup_runs": int(warmup_runs),
        "phase_order": phase_order,
        "scenario_count": len(paired_scenarios),
        "paired_run_count": len(paired_runs),
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_request_count,
        "gpu_memory_utilization": 0.50,
        "tokenizer_load_ms": tokenizer_load_ms,
        "prompt_format_ms": prompt_format_ms,
        "async_engine_load_ms": async_result["engine_load_ms"],
        "server_ready_ms": server_result["server_ready_ms"],
        "async_warmup": async_result["warmup"],
        "server_warmup": server_result["warmup"],
        "scenario_plan": plan_summary,
        "summary": _vllm_server_async_paired_summary_rows(paired_scenarios),
        "paired_runs": paired_runs,
        "paired_scenarios": paired_scenarios,
        "async_scenario_runs": async_result["scenario_runs"],
        "server_scenario_runs": server_result["scenario_runs"],
        "server_command": server_result["server_command"],
        "vllm_version": str(vllm.__version__),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "nvidia_smi_before": nvidia_smi_before,
        "nvidia_smi_after": nvidia_smi_after,
        "server_logs_head": server_result["server_logs_head"],
        "server_logs_tail": server_result["server_logs_tail"],
        "mean_server_to_async_throughput_ratio": _mean_present(
            row["server_to_async_output_tokens_per_second_ratio"] for row in paired_runs
        ),
        "mean_server_to_async_first_event_ratio": _mean_present(
            row["server_to_async_p95_first_event_ms_ratio"] for row in paired_runs
        ),
        "mean_server_to_async_latency_ratio": _mean_present(
            row["server_to_async_p95_latency_ms_ratio"] for row in paired_runs
        ),
        "mean_server_to_async_tpot_ratio": _mean_present(
            row["server_to_async_p95_stream_tpot_ms_ratio"] for row in paired_runs
        ),
        "note": (
            "One Modal worker runs in-process AsyncLLM and the OpenAI-compatible "
            f"vLLM server in phase order {phase_order} with the same scenario plan. "
            "Paired rows compare matching scenario_id and repeat_index."
        ),
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
    warmup_runs: int = 1,
    warmup_prompt_profile: str = "",
    phase_order: str = "async_first",
    scenario_seed: int = DEFAULT_VLLM_SWEEP_SEED,
    prefix_caching: str = "off",
    cache_metrics: str = "off",
    kv_cache_metrics_sample: float = 1.0,
    cold_sweep_dir: str = DEFAULT_VLLM_SWEEP_OUTPUT,
    prefix_sweep_dir: str = DEFAULT_VLLM_PREFIX_CACHE_SWEEP_OUTPUT,
    async_sweep_dir: str = DEFAULT_VLLM_SWEEP_OUTPUT,
    server_sweep_dir: str = DEFAULT_VLLM_SERVER_SWEEP_OUTPUT,
    async_first_paired_dir: str = DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT,
    server_first_paired_dir: str = DEFAULT_VLLM_SERVER_ASYNC_PAIRED_SERVER_FIRST_OUTPUT,
    phase_order_compare_dirs: str = DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_DIRS,
    short_multitrial_dir: str = DEFAULT_VLLM_SERVER_ASYNC_MULTITRIAL_OUTPUT,
    long_phase_order_compare_dir: str = DEFAULT_VLLM_SERVER_ASYNC_LONG_PHASE_ORDER_COMPARE_OUTPUT,
    prefix_cache_cold_first_paired_dir: str = DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT,
    prefix_cache_cache_first_paired_dir: str = DEFAULT_VLLM_PREFIX_CACHE_PAIRED_CACHE_FIRST_OUTPUT,
    prefix_cache_phase_order_compare_dir: str = DEFAULT_VLLM_PREFIX_CACHE_PHASE_ORDER_COMPARE_OUTPUT,
    prefix_cache_profile_control_dirs: str = DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_DIRS,
    prefix_cache_isolated_metrics_dir: str = DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_METRICS_OUTPUT,
    output_dir: str = "",
) -> None:
    if mode == "gpu-probe":
        payload = run_gpu_probe_remote.remote()
        output_path = Path(output_dir or DEFAULT_GPU_PROBE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "probe.json"
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

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

    if mode == "vllm-cache-metrics-probe":
        payload = run_vllm_cache_metrics_probe_remote.remote(hf_model=hf_model)
        output_path = Path(output_dir or DEFAULT_VLLM_CACHE_METRICS_PROBE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "cache-metrics-probe.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"vllm_version: {payload['vllm_version']}")
        print(
            "async_engine_args_relevant_parameters: "
            f"{','.join(payload['async_engine_args_relevant_parameters'])}"
        )
        print(
            "observability_config_relevant_parameters: "
            f"{','.join(payload['observability_config_relevant_parameters'])}"
        )
        for check in payload["constructor_checks"]:
            print(f"constructor_check {check['kwargs']}: accepted={check['accepted']}")
        print(f"json: {json_path}")
        return

    if mode == "vllm-cache-metrics-smoke":
        payload = run_vllm_cache_metrics_smoke_remote.remote(
            hf_model=hf_model,
            prompt_profile=prompt_profile,
            request_count=prompt_count,
            max_new_tokens=max_new_tokens,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_CACHE_METRICS_SMOKE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "cache-metrics-smoke.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        print(f"model: {payload['model_id']}")
        print(f"vllm_version: {payload['vllm_version']}")
        print(f"prompt_profile: {payload['prompt_profile']}")
        print(f"request_count: {payload['request_count']}")
        print(f"engine_load_ms: {payload['smoke']['engine_load_ms']:.3f}")
        for call in payload["smoke"]["log_stats_calls"]:
            print(
                f"log_stats {call['label']}: ok={call['ok']} "
                f"return_type={call['return_type']}"
            )
        print(
            "engine_relevant_attrs: "
            f"{len(payload['smoke']['engine_relevant_attrs'])}"
        )
        print(
            "engine_core_relevant_attrs: "
            f"{len(payload['smoke']['engine_core_relevant_attrs'])}"
        )
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

    if mode == "vllm-server-sweep":
        payload = run_vllm_server_sweep_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            warmup_runs=warmup_runs,
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_SWEEP_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "vllm-server-sweep.json"
        csv_path = output_path / "vllm-server-sweep.csv"
        run_csv_path = output_path / "vllm-server-sweep-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["run_summary"])

        print(f"model: {payload['model_id']}")
        print(f"backend: {payload['backend']} {payload['vllm_version']}")
        print(f"server_ready_ms: {payload['server_ready_ms']:.3f}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"scenario_runs: {payload['scenario_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"scenario_seed: {payload['scenario_seed']}")
        print(f"max_num_seqs: {payload['max_num_seqs']}")
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"runs_csv: {run_csv_path}")
        return

    if mode == "vllm-server-sweep-compare":
        payload = _compare_vllm_server_async_csvs(
            async_csv=Path(async_sweep_dir) / "vllm-sweep.csv",
            server_csv=Path(server_sweep_dir) / "vllm-server-sweep.csv",
            async_label="in_process_async_llm",
            server_label="openai_compatible_server",
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_SWEEP_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "server-vs-async.json"
        csv_path = output_path / "server-vs-async.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(
            "mean_server_to_async_throughput_ratio: "
            f"{payload['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "mean_server_to_async_latency_ratio: "
            f"{payload['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-paired":
        payload = run_vllm_server_async_paired_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            warmup_runs=warmup_runs,
            phase_order=phase_order,
        )
        phase_order_slug = phase_order.lower().replace("_", "-")
        default_output = (
            DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT
            if phase_order_slug == "async-first"
            else f"{DEFAULT_VLLM_SERVER_ASYNC_PAIRED_OUTPUT}-{phase_order_slug}"
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "paired-server-async.json"
        summary_csv_path = output_path / "paired-server-async-summary.csv"
        run_csv_path = output_path / "paired-server-async-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(summary_csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["paired_runs"])

        print(f"model: {payload['model_id']}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"paired_runs: {payload['paired_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"phase_order: {payload['phase_order']}")
        print(
            "mean_server_to_async_throughput_ratio: "
            f"{payload['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "mean_server_to_async_latency_ratio: "
            f"{payload['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"summary_csv: {summary_csv_path}")
        print(f"runs_csv: {run_csv_path}")
        return

    if mode == "vllm-server-async-phase-order-compare":
        payload = _compare_vllm_server_async_phase_orders(
            async_first_dir=Path(async_first_paired_dir),
            server_first_dir=Path(server_first_paired_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_ASYNC_PHASE_ORDER_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "phase-order-compare.json"
        csv_path = output_path / "phase-order-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(
            "async_first_mean_server_to_async_throughput_ratio: "
            f"{payload['async_first']['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "server_first_mean_server_to_async_throughput_ratio: "
            f"{payload['server_first']['mean_server_to_async_throughput_ratio']:.3f}"
        )
        print(
            "async_first_mean_server_to_async_latency_ratio: "
            f"{payload['async_first']['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(
            "server_first_mean_server_to_async_latency_ratio: "
            f"{payload['server_first']['mean_server_to_async_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-multitrial-aggregate":
        payload = _aggregate_vllm_server_async_phase_order_trials(
            compare_dirs=[
                Path(path)
                for path in _split_csv(phase_order_compare_dirs)
            ],
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_ASYNC_MULTITRIAL_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "phase-order-multitrial.json"
        csv_path = output_path / "phase-order-multitrial.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])

        print(f"trials: {payload['trial_count']}")
        for row in payload["summary"]:
            print(
                f"{row['metric']}: async_first_mean={row['async_first_mean']:.3f} "
                f"server_first_mean={row['server_first_mean']:.3f} "
                f"delta_mean={row['server_first_minus_async_first_mean']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-server-async-workload-compare":
        payload = _compare_vllm_server_async_workload_profiles(
            short_multitrial_dir=Path(short_multitrial_dir),
            long_phase_order_compare_dir=Path(long_phase_order_compare_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_SERVER_ASYNC_WORKLOAD_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "workload-compare.json"
        csv_path = output_path / "workload-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"metrics: {payload['metric_count']}")
        print(f"short_trials: {payload['short_trial_count']}")
        print(f"long_paired_runs: {payload['long_paired_run_count']}")
        for row in payload["rows"]:
            print(
                f"{row['metric']}: short_delta_mean="
                f"{row['short_order_effect_delta_mean']:.3f} long_delta_mean="
                f"{row['long_order_effect_delta_mean']:.3f} "
                f"long_minus_short_delta_mean="
                f"{row['long_minus_short_order_effect_delta_mean']:.3f}"
            )
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

    if mode == "vllm-prefix-cache-isolated-window-summary":
        payload = _summarize_vllm_prefix_cache_isolated_window(
            isolated_metrics_dir=Path(prefix_cache_isolated_metrics_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WINDOW_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-window-summary.json"
        csv_path = output_path / "prefix-cache-window-summary.csv"
        profile_csv_path = output_path / "prefix-cache-window-profile-control.csv"
        markdown_path = output_path / "prefix-cache-window-summary.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["scenario_rows"])
        _write_records_csv(profile_csv_path, payload["profile_control_rows"])
        markdown_path.write_text(payload["markdown"], encoding="utf-8")

        print(f"scenario_rows: {payload['scenario_count']}")
        print(f"profile_control_rows: {payload['profile_control_row_count']}")
        print(
            "mean_shared_minus_control_estimated_measured_cache_hit_rate_pct: "
            f"{payload['summary']['mean_shared_minus_control_estimated_measured_cache_hit_rate_pct']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        print(f"markdown: {markdown_path}")
        return

    if mode == "vllm-prefix-cache-isolated-stability-summary":
        payload = _summarize_vllm_prefix_cache_isolated_stability(
            isolated_metrics_dir=Path(prefix_cache_isolated_metrics_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_STABILITY_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-isolated-stability-summary.json"
        csv_path = output_path / "prefix-cache-isolated-stability-summary.csv"
        profile_csv_path = output_path / "prefix-cache-isolated-stability-profile-control.csv"
        markdown_path = output_path / "prefix-cache-isolated-stability-summary.md"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["scenario_rows"])
        _write_records_csv(profile_csv_path, payload["profile_control_rows"])
        markdown_path.write_text(payload["markdown"], encoding="utf-8")

        print(f"scenario_rows: {payload['scenario_count']}")
        print(f"profile_control_rows: {payload['profile_control_row_count']}")
        print(
            "mean_shared_minus_control_cache_hit_rate_pct: "
            f"{payload['summary']['mean_shared_minus_control_cache_hit_rate_pct']:.3f}"
        )
        print(
            "max_cache_hit_rate_population_stdev: "
            f"{payload['summary']['max_cache_hit_rate_population_stdev']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        print(f"markdown: {markdown_path}")
        return

    if mode in {
        "vllm-prefix-cache-isolated-metrics",
        "vllm-prefix-cache-isolated-warm-window",
        "vllm-prefix-cache-isolated-neutral-warmup",
    }:
        request_count_values = _split_positive_int_csv(request_counts, "request_counts")
        prompt_profile_values = [
            profile.lower().replace("-", "_")
            for profile in _split_csv(prompt_profiles)
        ]
        output_token_values = _split_positive_int_csv(output_tokens, "output_tokens")
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        _validate_vllm_prompt_profiles(prompt_profile_values)
        isolated_phase_order = phase_order.lower().replace("-", "_")
        if isolated_phase_order not in {"cold_first", "cache_first"}:
            isolated_phase_order = "cold_first"
        effective_warmup_runs = (
            1
            if mode
            in {
                "vllm-prefix-cache-isolated-warm-window",
                "vllm-prefix-cache-isolated-neutral-warmup",
            }
            else 0
        )
        effective_warmup_prompt_profile = (
            warmup_prompt_profile.strip().lower().replace("-", "_")
            if warmup_prompt_profile.strip()
            else "neutral_long"
            if mode == "vllm-prefix-cache-isolated-neutral-warmup"
            else ""
        )

        paired_runs = []
        remote_call_summaries = []
        call_index = 0
        for repeat_index in range(repeats):
            for prompt_profile_value in prompt_profile_values:
                for max_tokens in output_token_values:
                    for request_count_value in request_count_values:
                        single_payload = run_vllm_prefix_cache_paired_remote.remote(
                            hf_model=hf_model,
                            request_counts=str(request_count_value),
                            prompt_profiles=prompt_profile_value,
                            output_tokens=str(max_tokens),
                            repeats=1,
                            scenario_seed=scenario_seed + repeat_index,
                            warmup_runs=effective_warmup_runs,
                            warmup_prompt_profile=effective_warmup_prompt_profile,
                            phase_order=isolated_phase_order,
                            collect_cache_metrics=True,
                            kv_cache_metrics_sample=kv_cache_metrics_sample,
                        )
                        if single_payload["paired_run_count"] != 1:
                            raise ValueError(
                                "Expected one paired run from isolated metrics call"
                            )
                        row = dict(single_payload["paired_runs"][0])
                        row["repeat_index"] = repeat_index
                        row["pair_id"] = (
                            f"{row['scenario_id']}_isolated_rep{repeat_index:02d}"
                        )
                        row["isolation_call_index"] = call_index
                        row["isolation_phase_order"] = single_payload["phase_order"]
                        row["isolation_warmup_runs"] = effective_warmup_runs
                        row["isolation_warmup_prompt_profile"] = (
                            effective_warmup_prompt_profile or None
                        )
                        paired_runs.append(row)

                        cold_run = single_payload["cold_scenario_runs"][0]
                        cache_run = single_payload["cache_scenario_runs"][0]
                        remote_call_summaries.append(
                            {
                                "isolation_call_index": call_index,
                                "repeat_index": repeat_index,
                                "scenario_id": row["scenario_id"],
                                "prompt_profile": row["prompt_profile"],
                                "request_count": row["request_count"],
                                "max_new_tokens": row["max_new_tokens"],
                                "phase_order": single_payload["phase_order"],
                                "warmup_prompt_profile": single_payload[
                                    "warmup_prompt_profile"
                                ],
                                "warmup_scenario_count": single_payload[
                                    "warmup_scenario_count"
                                ],
                                "cold_engine_load_ms": single_payload[
                                    "cold_engine_load_ms"
                                ],
                                "cache_engine_load_ms": single_payload[
                                    "cache_engine_load_ms"
                                ],
                                "cold_prefix_cache_hit_rate_pct": row[
                                    "cold_prefix_cache_hit_rate_pct"
                                ],
                                "cache_prefix_cache_hit_rate_pct": row[
                                    "cache_prefix_cache_hit_rate_pct"
                                ],
                                "cache_to_cold_prefix_cache_hit_rate_pct_delta": row[
                                    "cache_to_cold_prefix_cache_hit_rate_pct_delta"
                                ],
                                "cache_to_cold_output_tokens_per_second_ratio": row[
                                    "cache_to_cold_output_tokens_per_second_ratio"
                                ],
                                "cache_to_cold_p95_latency_ms_ratio": row[
                                    "cache_to_cold_p95_latency_ms_ratio"
                                ],
                                "warmup_runs": effective_warmup_runs,
                                "cold_warmup": single_payload["cold_warmup"],
                                "cache_warmup": single_payload["cache_warmup"],
                                "cold_warmup_cache_metrics": single_payload[
                                    "cold_warmup_cache_metrics"
                                ],
                                "cache_warmup_cache_metrics": single_payload[
                                    "cache_warmup_cache_metrics"
                                ],
                                "cold_warmup_cache_metrics_log_stats": single_payload[
                                    "cold_warmup_cache_metrics_log_stats"
                                ],
                                "cache_warmup_cache_metrics_log_stats": single_payload[
                                    "cache_warmup_cache_metrics_log_stats"
                                ],
                                "cold_cache_metrics": cold_run.get("cache_metrics"),
                                "cache_cache_metrics": cache_run.get("cache_metrics"),
                                "cold_cache_metrics_log_stats": cold_run.get(
                                    "cache_metrics_log_stats"
                                ),
                                "cache_cache_metrics_log_stats": cache_run.get(
                                    "cache_metrics_log_stats"
                                ),
                            }
                        )
                        call_index += 1

        paired_scenarios = _aggregate_vllm_prefix_cache_paired_scenarios(paired_runs)
        summary_rows = _vllm_prefix_cache_paired_summary_rows(paired_scenarios)
        profile_control_rows = _vllm_prefix_cache_isolated_profile_control_rows(
            paired_scenarios
        )
        payload = {
            "schema_version": 1,
            "execution": "modal",
            "mode": mode,
            "backend": "vllm-paired-prefix-cache",
            "model_id": hf_model,
            "request_counts": request_count_values,
            "prompt_profiles": prompt_profile_values,
            "output_tokens": output_token_values,
            "repeats": int(repeats),
            "scenario_seed": int(scenario_seed),
            "phase_order": isolated_phase_order,
            "warmup_runs": effective_warmup_runs,
            "warmup_prompt_profile": effective_warmup_prompt_profile or None,
            "collect_cache_metrics": True,
            "kv_cache_metrics_sample": float(kv_cache_metrics_sample),
            "isolation_method": (
                "Each scenario/repeat is executed by a separate remote paired "
                "benchmark call with fresh cold/cache AsyncLLM engines. The "
                f"run uses {effective_warmup_runs} warmup scenario run(s) "
                "inside each fresh engine before the measured scenario. "
                f"Warmup prompt profile: {effective_warmup_prompt_profile or 'measured scenario'}."
            ),
            "remote_call_count": len(remote_call_summaries),
            "scenario_count": len(paired_scenarios),
            "paired_run_count": len(paired_runs),
            "summary": summary_rows,
            "paired_runs": paired_runs,
            "paired_scenarios": paired_scenarios,
            "profile_control": {
                "shared_profile": "shared_prefix_long",
                "control_profile": "matched_unique_prefix",
                "row_count": len(profile_control_rows),
                "summary": _vllm_prefix_cache_isolated_profile_control_summary(
                    profile_control_rows
                ),
                "rows": profile_control_rows,
            },
            "remote_call_summaries": remote_call_summaries,
            "mean_cache_to_cold_throughput_ratio": _mean_present(
                row["cache_to_cold_output_tokens_per_second_ratio"]
                for row in paired_runs
            ),
            "mean_cache_to_cold_first_event_ratio": _mean_present(
                row["cache_to_cold_p95_first_event_ms_ratio"] for row in paired_runs
            ),
            "mean_cache_to_cold_latency_ratio": _mean_present(
                row["cache_to_cold_p95_latency_ms_ratio"] for row in paired_runs
            ),
            "mean_cache_to_cold_tpot_ratio": _mean_present(
                row["cache_to_cold_p95_stream_tpot_ms_ratio"] for row in paired_runs
            ),
            "note": (
                "This is a metric-isolation harness. The warm-window mode adds "
                "a throwaway warmup window before the measured scenario; the "
                "default isolated mode runs with no warmup."
            ),
        }
        default_output = (
            DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_WARM_WINDOW_OUTPUT
            if mode == "vllm-prefix-cache-isolated-warm-window"
            else DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_NEUTRAL_WARMUP_OUTPUT
            if mode == "vllm-prefix-cache-isolated-neutral-warmup"
            else DEFAULT_VLLM_PREFIX_CACHE_ISOLATED_METRICS_OUTPUT
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-isolated-metrics.json"
        summary_csv_path = output_path / "prefix-cache-isolated-metrics-summary.csv"
        runs_csv_path = output_path / "prefix-cache-isolated-metrics-runs.csv"
        profile_csv_path = output_path / "prefix-cache-isolated-profile-control.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(summary_csv_path, payload["summary"])
        _write_records_csv(runs_csv_path, payload["paired_runs"])
        _write_records_csv(profile_csv_path, profile_control_rows)

        print(f"scenarios: {payload['scenario_count']}")
        print(f"paired_runs: {payload['paired_run_count']}")
        print(f"remote_calls: {payload['remote_call_count']}")
        print(f"phase_order: {payload['phase_order']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"warmup_prompt_profile: {payload['warmup_prompt_profile']}")
        print(
            "mean_cache_to_cold_throughput_ratio: "
            f"{payload['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "mean_cache_to_cold_latency_ratio: "
            f"{payload['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        profile_summary = payload["profile_control"]["summary"]
        if profile_summary:
            print(
                "mean_shared_minus_control_cache_hit_rate_delta: "
                f"{profile_summary['mean_shared_minus_control_cache_prefix_cache_hit_rate_pct_median']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"summary_csv: {summary_csv_path}")
        print(f"runs_csv: {runs_csv_path}")
        print(f"profile_csv: {profile_csv_path}")
        return

    if mode == "vllm-prefix-cache-paired":
        collect_cache_metrics = _parse_bool_choice(cache_metrics, "cache_metrics")
        payload = run_vllm_prefix_cache_paired_remote.remote(
            hf_model=hf_model,
            request_counts=request_counts,
            prompt_profiles=prompt_profiles,
            output_tokens=output_tokens,
            repeats=repeats,
            scenario_seed=scenario_seed,
            warmup_runs=warmup_runs,
            warmup_prompt_profile=warmup_prompt_profile,
            phase_order=phase_order,
            collect_cache_metrics=collect_cache_metrics,
            kv_cache_metrics_sample=kv_cache_metrics_sample,
        )
        phase_order_slug = payload["phase_order"].lower().replace("_", "-")
        default_output = (
            DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT
            if phase_order_slug == "cold-first"
            else f"{DEFAULT_VLLM_PREFIX_CACHE_PAIRED_OUTPUT}-{phase_order_slug}"
        )
        output_path = Path(output_dir or default_output)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "paired-prefix-cache.json"
        summary_csv_path = output_path / "paired-prefix-cache-summary.csv"
        run_csv_path = output_path / "paired-prefix-cache-runs.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(summary_csv_path, payload["summary"])
        _write_records_csv(run_csv_path, payload["paired_runs"])

        print(f"model: {payload['model_id']}")
        print(f"scenarios: {payload['scenario_count']}")
        print(f"paired_runs: {payload['paired_run_count']}")
        print(f"repeats: {payload['repeats']}")
        print(f"warmup_runs: {payload['warmup_runs']}")
        print(f"warmup_prompt_profile: {payload['warmup_prompt_profile']}")
        print(f"phase_order: {payload['phase_order']}")
        print(f"collect_cache_metrics: {payload['collect_cache_metrics']}")
        print(
            "mean_cache_to_cold_throughput_ratio: "
            f"{payload['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "mean_cache_to_cold_latency_ratio: "
            f"{payload['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"summary_csv: {summary_csv_path}")
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

    if mode == "vllm-prefix-cache-phase-order-compare":
        payload = _compare_vllm_prefix_cache_phase_orders(
            cold_first_dir=Path(prefix_cache_cold_first_paired_dir),
            cache_first_dir=Path(prefix_cache_cache_first_paired_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_PHASE_ORDER_COMPARE_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-phase-order-compare.json"
        csv_path = output_path / "prefix-cache-phase-order-compare.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"scenarios: {payload['scenario_count']}")
        print(
            "cold_first_mean_cache_to_cold_throughput_ratio: "
            f"{payload['cold_first']['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "cache_first_mean_cache_to_cold_throughput_ratio: "
            f"{payload['cache_first']['mean_cache_to_cold_throughput_ratio']:.3f}"
        )
        print(
            "cold_first_mean_cache_to_cold_latency_ratio: "
            f"{payload['cold_first']['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        print(
            "cache_first_mean_cache_to_cold_latency_ratio: "
            f"{payload['cache_first']['mean_cache_to_cold_latency_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-prefix-cache-profile-control":
        payload = _compare_vllm_prefix_cache_profile_controls(
            phase_order_compare_dir=Path(prefix_cache_phase_order_compare_dir),
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_PROFILE_CONTROL_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-profile-control.json"
        csv_path = output_path / "prefix-cache-profile-control.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["rows"])

        print(f"profile_control_rows: {payload['row_count']}")
        print(
            "mean_shared_minus_control_cold_first_throughput_ratio: "
            f"{payload['summary']['mean_shared_minus_control_cold_first_output_tokens_per_second_ratio']:.3f}"
        )
        print(
            "mean_shared_minus_control_cache_first_throughput_ratio: "
            f"{payload['summary']['mean_shared_minus_control_cache_first_output_tokens_per_second_ratio']:.3f}"
        )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode == "vllm-prefix-cache-profile-multitrial":
        payload = _aggregate_vllm_prefix_cache_profile_control_trials(
            profile_control_dirs=[
                Path(path)
                for path in _split_csv(prefix_cache_profile_control_dirs)
            ],
        )
        output_path = Path(output_dir or DEFAULT_VLLM_PREFIX_CACHE_PROFILE_MULTITRIAL_OUTPUT)
        output_path.mkdir(parents=True, exist_ok=True)
        json_path = output_path / "prefix-cache-profile-multitrial.json"
        csv_path = output_path / "prefix-cache-profile-multitrial.csv"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_records_csv(csv_path, payload["summary"])

        print(f"trials: {payload['trial_count']}")
        for row in payload["summary"]:
            print(
                f"{row['phase_order']} {row['metric']}: "
                f"mean_delta={row['shared_minus_control_mean']:.3f} "
                f"min={row['shared_minus_control_min']:.3f} "
                f"max={row['shared_minus_control_max']:.3f}"
            )
        print(f"json: {json_path}")
        print(f"csv: {csv_path}")
        return

    if mode != "sweep":
        raise ValueError(
            "mode must be 'sweep', 'gpu-probe', 'tiny-inference', "
            "'vllm-cache-metrics-probe', 'vllm-inference', "
            "'vllm-cache-metrics-smoke', 'vllm-streaming', 'vllm-concurrent', "
            "'vllm-server-streaming', 'vllm-server-concurrent', "
            "'vllm-server-sweep', 'vllm-server-sweep-compare', "
            "'vllm-server-async-paired', "
            "'vllm-server-async-phase-order-compare', "
            "'vllm-server-async-multitrial-aggregate', "
            "'vllm-server-async-workload-compare', 'vllm-sweep', or "
            "'vllm-prefix-cache-isolated-window-summary', "
            "'vllm-prefix-cache-isolated-stability-summary', "
            "'vllm-prefix-cache-isolated-neutral-warmup', "
            "'vllm-prefix-cache-isolated-warm-window', "
            "'vllm-prefix-cache-isolated-metrics', "
            "'vllm-prefix-cache-paired', 'vllm-prefix-cache-compare', or "
            "'vllm-prefix-cache-phase-order-compare', or "
            "'vllm-prefix-cache-profile-control', or "
            "'vllm-prefix-cache-profile-multitrial'"
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


def _validate_vllm_prompt_profiles(profiles: list[str], label: str = "prompt_profiles") -> None:
    invalid_profiles = [
        profile for profile in profiles
        if profile not in VALID_VLLM_PROMPT_PROFILES
    ]
    if invalid_profiles:
        valid_profiles = ", ".join(sorted(VALID_VLLM_PROMPT_PROFILES))
        invalid_list = ", ".join(invalid_profiles)
        raise ValueError(
            f"{label} contains unsupported profiles: {invalid_list}. "
            f"Valid profiles: {valid_profiles}"
        )


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
    prompt_profile = prompt_profile.lower().replace("-", "_")
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
    if prompt_profile == "shared_prefix":
        return _select_shared_prefix_prompts(prompt_count)
    if prompt_profile == "shared_prefix_long":
        return _select_shared_prefix_prompts(prompt_count, long_context=True)
    if prompt_profile == "matched_unique_prefix":
        return _select_matched_unique_prefix_prompts(prompt_count)
    if prompt_profile == "neutral_long":
        return _select_neutral_long_prompts(prompt_count)
    valid_profiles = ", ".join(sorted(VALID_VLLM_PROMPT_PROFILES))
    raise ValueError(f"prompt_profile must be one of: {valid_profiles}")


def _select_shared_prefix_prompts(
    prompt_count: int,
    long_context: bool = False,
) -> list[str]:
    common_prefix = (
        "You are evaluating a GPU inference platform during a capacity incident. "
        "Every request in this batch shares the same incident brief, model "
        "deployment notes, and operational constraints. The service runs an "
        "OpenAI-compatible endpoint backed by vLLM on one memory-constrained GPU. "
        "Traffic arrives in bursts, responses stream to users, prefill competes "
        "with decode for scheduler time, and the KV cache can dominate available "
        "memory when many long contexts overlap. Operators care about p95 time to "
        "first token, p95 end-to-end latency, output tokens per second, and the "
        "amount of reusable prefix work. The common incident facts are: model "
        "weights are already resident, requests use deterministic decoding, "
        "prefix cache blocks can be reused only when leading prompt tokens match "
        "exactly, and request-specific details appear only after this shared "
        "brief. Use the shared context as ground truth and answer the final task."
    )
    if long_context:
        common_prefix = f"{common_prefix}\n\n{_long_prefix_cache_context()}"
    suffixes = [
        "Task A: explain how prefix caching changes prefill cost for this batch.",
        "Task B: identify which metric should move first if KV reuse is effective.",
        "Task C: describe the failure mode if cache blocks fragment under load.",
        "Task D: compare throughput impact at one, two, four, and eight requests.",
        "Task E: explain why decode may dominate after prefill is avoided.",
        "Task F: name one measurement that would separate cache reuse from noise.",
        "Task G: summarize how phase order could confound this experiment.",
        "Task H: recommend the next benchmark to validate the observed effect.",
    ]
    prompts = []
    for index in range(prompt_count):
        suffix = suffixes[index % len(suffixes)]
        prompts.append(f"{common_prefix}\n\n{suffix}")
    return prompts


def _select_matched_unique_prefix_prompts(prompt_count: int) -> list[str]:
    suffixes = [
        "Task A: explain how prefix caching changes prefill cost for this batch.",
        "Task B: identify which metric should move first if KV reuse is effective.",
        "Task C: describe the failure mode if cache blocks fragment under load.",
        "Task D: compare throughput impact at one, two, four, and eight requests.",
        "Task E: explain why decode may dominate after prefill is avoided.",
        "Task F: name one measurement that would separate cache reuse from noise.",
        "Task G: summarize how phase order could confound this experiment.",
        "Task H: recommend the next benchmark to validate the observed effect.",
    ]
    prompts = []
    for index in range(prompt_count):
        suffix = suffixes[index % len(suffixes)]
        unique_context = _matched_unique_prefix_context(index)
        prompts.append(f"{unique_context}\n\n{suffix}")
    return prompts


def _select_neutral_long_prompts(prompt_count: int) -> list[str]:
    suffixes = [
        "Calibration task A: classify which queue should receive this packet.",
        "Calibration task B: summarize the dominant constraint in one sentence.",
        "Calibration task C: identify one field that would be useful to log.",
        "Calibration task D: compare the packet with a steady-state request.",
        "Calibration task E: explain how admission order affects the report.",
        "Calibration task F: name one sanity check for the measurement window.",
        "Calibration task G: describe the expected scheduler bookkeeping.",
        "Calibration task H: recommend the next diagnostic counter to inspect.",
    ]
    prompts = []
    for index in range(prompt_count):
        suffix = suffixes[index % len(suffixes)]
        unique_context = _neutral_long_context(index)
        prompts.append(f"{unique_context}\n\n{suffix}")
    return prompts


def _long_prefix_cache_context() -> str:
    return (
        "Extended incident packet. The workload trace contains one warm model, "
        "one tokenizer, a fixed CUDA graph policy, and an arrival burst where "
        "all user requests begin with the same runbook, tenant limits, model "
        "deployment notes, and safety constraints. The platform team suspects "
        "prefill dominates early latency because every prompt repeats the same "
        "policy and trace summary before reaching a short request-specific task. "
        "The scheduler admits all requests at nearly the same time, so reusable "
        "prefix blocks should reduce repeated attention work during prefill while "
        "leaving decode token generation mostly unchanged.\n\n"
        "Cache mechanics. Prefix-cache reuse is only valid when leading tokens "
        "match exactly through complete cache blocks. A small edit near the front "
        "of the prompt should destroy most reuse, while a long identical incident "
        "brief should allow later requests to skip repeated prefix computation "
        "after blocks are populated. This benchmark therefore watches first-token "
        "latency, end-to-end p95 latency, output tokens per second, and time per "
        "output token. A true cache effect should be strongest when multiple "
        "concurrent requests share the same long leading context.\n\n"
        "Operational constraints. The GPU is a memory-constrained T4, the model "
        "is small enough to run cheaply but still exercises vLLM scheduling, "
        "decoding is deterministic, output lengths are fixed by scenario, and "
        "the paired experiment runs prefix caching off and on inside one Modal "
        "worker. The evidence should separate prefix reuse from warm-engine "
        "effects, phase order, prompt length, and random service noise.\n\n"
        "Decision rule. If the shared-prefix profile improves with caching while "
        "the matched unique-prefix profile does not, the result supports a real "
        "prefix-cache claim. If both profiles improve similarly, the benchmark is "
        "probably measuring warm state or shape effects instead of cache reuse."
    )


def _matched_unique_prefix_context(index: int) -> str:
    labels = [
        "Alpha-17",
        "Bravo-26",
        "Crimson-35",
        "Delta-44",
        "Ember-53",
        "Falcon-62",
        "Graphite-71",
        "Harbor-80",
    ]
    label = labels[index % len(labels)]
    return (
        f"{label} incident packet. This request intentionally starts with a "
        "different leading identifier, tenant trace, service owner, and rollout "
        "history so the large reusable prefix from the shared-context workload "
        "is not present. The workload still contains one warm model, one "
        "tokenizer, a fixed CUDA graph policy, and an arrival burst where user "
        "requests have comparable runbooks, tenant limits, model deployment "
        "notes, and safety constraints. The platform team suspects prefill "
        "dominates early latency because each prompt carries a long operational "
        "brief before reaching a short request-specific task. The scheduler "
        "admits all requests at nearly the same time, but the leading tokens "
        "diverge before a reusable cache block can form.\n\n"
        "Control mechanics. Prefix-cache reuse is only valid when leading tokens "
        "match exactly through complete cache blocks. This control keeps the "
        "prompt length, topic, vocabulary, and requested output shape close to "
        "the shared-prefix case while changing the beginning of every prompt. "
        "The benchmark therefore watches first-token latency, end-to-end p95 "
        "latency, output tokens per second, and time per output token. A true "
        "cache effect should be weaker here than in the shared-prefix profile.\n\n"
        "Operational constraints. The GPU is a memory-constrained T4, the model "
        "is small enough to run cheaply but still exercises vLLM scheduling, "
        "decoding is deterministic, output lengths are fixed by scenario, and "
        "the paired experiment runs prefix caching off and on inside one Modal "
        "worker. The evidence should separate prefix reuse from warm-engine "
        "effects, phase order, prompt length, and random service noise.\n\n"
        "Synthetic trace detail. The tenant has eight queued requests, each "
        "request references a different dashboard, rollout window, service "
        "owner, alert identifier, and mitigation note, and each request carries "
        "a separate prefill-heavy runbook. The content is intentionally similar "
        "in topic and size to the shared-prefix workload, but it is arranged so "
        "the beginning of every prompt diverges immediately. The benchmark "
        "should therefore preserve comparable context length, vocabulary, output "
        "budget, scheduler pressure, and decode work without providing a long "
        "identical leading block for vLLM to reuse. Any remaining speedup in "
        "this control is more likely to be phase order, warm allocation state, "
        "or generic shape reuse than exact-prefix KV-cache reuse. The control "
        "also keeps the same measurement vocabulary around prefill, decode, "
        "tail latency, throughput, cache blocks, and paired phase ordering.\n\n"
        "Decision rule. If this unique-prefix control improves by the same amount "
        "as the shared-prefix profile, the result is probably not caused by "
        "large exact-prefix reuse. If this control is flat while shared-prefix "
        "improves, the benchmark has a stronger KV-cache signal."
    )


def _neutral_long_context(index: int) -> str:
    labels = [
        "Quartz-11",
        "Lumen-22",
        "Atlas-33",
        "Nimbus-44",
        "Vector-55",
        "Solace-66",
        "Orchid-77",
        "Zenith-88",
    ]
    label = labels[index % len(labels)]
    return (
        f"{label} calibration packet. This warmup request is deliberately "
        "unrelated to the measured incident profiles. It begins with a unique "
        "marker, a separate routing note, a different document title, and an "
        "independent checklist so it should not seed the measured prefix-cache "
        "state beyond short tokenizer and chat-template scaffolding. The packet "
        "is only meant to exercise a comparable long-prefill request shape.\n\n"
        "Calibration background. The synthetic document describes a nightly "
        "data-quality audit for a batch analytics service. The audit tracks "
        "schema drift, missing partitions, retry budgets, timestamp skew, "
        "checksum mismatches, and dashboard refresh order. The worker pool has "
        "already loaded its dependencies, the input tables are fixed for the "
        "trial, and every request asks for a deterministic short answer. The "
        "content is verbose on purpose so the serving engine allocates and "
        "schedules a long prompt before producing only a small number of output "
        "tokens.\n\n"
        "Shape notes. Each calibration request keeps a similar envelope: one "
        "long context, one short task, one deterministic decode setting, and a "
        "batch admitted at nearly the same time. The text avoids the measured "
        "shared incident brief, tenant rollout language, prefix-cache decision "
        "rule, and benchmark-specific control paragraphs. Its job is to touch "
        "the same general prefill and decode machinery while avoiding the exact "
        "leading token sequences that the measured cache experiment evaluates.\n\n"
        "Audit details. The batch has eight possible records, each record uses "
        "a separate source table, owner alias, retry window, freshness target, "
        "and downstream report. The reviewer must decide whether the record is "
        "ready for publication, should be quarantined for another validation "
        "pass, or should be escalated because a threshold changed. The details "
        "are intentionally repetitive in structure but different in leading "
        "tokens, which keeps the warmup useful for request-shape coverage "
        "without constructing a large common prefix for the measured workload.\n\n"
        "Calibration rule. Treat this packet as a warmup-only document. The "
        "answer should be concise, deterministic, and based only on the audit "
        "fields in this packet. Do not assume it shares state, document text, "
        "or reusable leading context with any later measured request."
    )


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


def _summarize_server_stream_requests(
    requests: list[dict[str, Any]],
    batch_wall_ms: float,
) -> dict[str, Any]:
    total_output_tokens = sum(request["generated_tokens"] for request in requests)
    first_contents = [
        request["first_content_ms"]
        for request in requests
        if request["first_content_ms"] is not None
    ]
    latencies = [request["request_wall_ms"] for request in requests]
    tpot_values = [
        request["stream_tpot_ms"]
        for request in requests
        if request["stream_tpot_ms"] is not None
    ]
    return {
        "batch_wall_ms": batch_wall_ms,
        "total_output_tokens": total_output_tokens,
        "aggregate_output_tokens_per_second": total_output_tokens / max(batch_wall_ms / 1000, 1e-9),
        "p50_first_content_ms": _percentile(first_contents, 50),
        "p95_first_content_ms": _percentile(first_contents, 95),
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


def _aggregate_vllm_server_scenarios(
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
            "p50_first_content_ms",
            "p95_first_content_ms",
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


def _vllm_server_sweep_summary_rows(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        "p95_first_content_ms_median",
        "p95_first_content_ms_p95",
        "p95_first_content_ms_min",
        "p95_first_content_ms_max",
        "p95_first_content_ms_cv",
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


def _vllm_server_sweep_run_rows(scenario_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
        "p50_first_content_ms",
        "p95_first_content_ms",
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
        f"{field}_cv": stddev / abs(mean_value) if mean_value else None,
    }


def _compare_vllm_server_async_csvs(
    async_csv: Path,
    server_csv: Path,
    async_label: str,
    server_label: str,
) -> dict[str, Any]:
    async_rows = _read_csv_by_key(async_csv, "scenario_id")
    server_rows = _read_csv_by_key(server_csv, "scenario_id")
    scenario_ids = sorted(set(async_rows) & set(server_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for comparison")

    rows = []
    for scenario_id in scenario_ids:
        async_row = async_rows[scenario_id]
        server_row = server_rows[scenario_id]

        async_throughput = _float_field(
            async_row,
            "aggregate_output_tokens_per_second_median",
        )
        server_throughput = _float_field(
            server_row,
            "aggregate_output_tokens_per_second_median",
        )
        async_first_event = _float_field(async_row, "p95_first_chunk_ms_median")
        server_first_event = _float_field(server_row, "p95_first_content_ms_median")
        async_latency = _float_field(async_row, "p95_latency_ms_median")
        server_latency = _float_field(server_row, "p95_latency_ms_median")
        async_tpot = _float_field(async_row, "p95_stream_tpot_ms_median")
        server_tpot = _float_field(server_row, "p95_stream_tpot_ms_median")
        async_batch_wall = _float_field(async_row, "batch_wall_ms_median")
        server_batch_wall = _float_field(server_row, "batch_wall_ms_median")

        rows.append(
            {
                "scenario_id": scenario_id,
                "prompt_profile": async_row["prompt_profile"],
                "request_count": int(async_row["request_count"]),
                "max_new_tokens": int(async_row["max_new_tokens"]),
                "async_repeats": int(async_row["repeats"]),
                "server_repeats": int(server_row["repeats"]),
                "async_prompt_tokens_mean": _float_field(async_row, "prompt_tokens_mean"),
                "server_prompt_tokens_mean": _float_field(server_row, "prompt_tokens_mean"),
                "async_output_tokens_per_second_median": async_throughput,
                "server_output_tokens_per_second_median": server_throughput,
                "server_to_async_output_tokens_per_second_delta": _delta(
                    server_throughput,
                    async_throughput,
                ),
                "server_to_async_output_tokens_per_second_ratio": _ratio(
                    server_throughput,
                    async_throughput,
                ),
                "async_p95_first_event_ms_median": async_first_event,
                "server_p95_first_content_ms_median": server_first_event,
                "server_to_async_p95_first_event_ms_delta": _delta(
                    server_first_event,
                    async_first_event,
                ),
                "server_to_async_p95_first_event_ms_ratio": _ratio(
                    server_first_event,
                    async_first_event,
                ),
                "async_p95_latency_ms_median": async_latency,
                "server_p95_latency_ms_median": server_latency,
                "server_to_async_p95_latency_ms_delta": _delta(
                    server_latency,
                    async_latency,
                ),
                "server_to_async_p95_latency_ms_ratio": _ratio(
                    server_latency,
                    async_latency,
                ),
                "async_p95_stream_tpot_ms_median": async_tpot,
                "server_p95_stream_tpot_ms_median": server_tpot,
                "server_to_async_p95_stream_tpot_ms_delta": _delta(
                    server_tpot,
                    async_tpot,
                ),
                "server_to_async_p95_stream_tpot_ms_ratio": _ratio(
                    server_tpot,
                    async_tpot,
                ),
                "async_batch_wall_ms_median": async_batch_wall,
                "server_batch_wall_ms_median": server_batch_wall,
                "server_to_async_batch_wall_ms_delta": _delta(
                    server_batch_wall,
                    async_batch_wall,
                ),
                "server_to_async_batch_wall_ms_ratio": _ratio(
                    server_batch_wall,
                    async_batch_wall,
                ),
                "async_throughput_cv": _float_field(
                    async_row,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "server_throughput_cv": _float_field(
                    server_row,
                    "aggregate_output_tokens_per_second_cv",
                ),
                "async_latency_cv": _float_field(async_row, "p95_latency_ms_cv"),
                "server_latency_cv": _float_field(server_row, "p95_latency_ms_cv"),
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-sweep-compare",
        "async_label": async_label,
        "server_label": server_label,
        "async_csv": str(async_csv),
        "server_csv": str(server_csv),
        "scenario_count": len(rows),
        "mean_server_to_async_throughput_ratio": _mean_present(
            row["server_to_async_output_tokens_per_second_ratio"] for row in rows
        ),
        "mean_server_to_async_first_event_ratio": _mean_present(
            row["server_to_async_p95_first_event_ms_ratio"] for row in rows
        ),
        "mean_server_to_async_latency_ratio": _mean_present(
            row["server_to_async_p95_latency_ms_ratio"] for row in rows
        ),
        "mean_server_to_async_tpot_ratio": _mean_present(
            row["server_to_async_p95_stream_tpot_ms_ratio"] for row in rows
        ),
        "rows": rows,
    }


def _compare_vllm_server_async_phase_orders(
    async_first_dir: Path,
    server_first_dir: Path,
) -> dict[str, Any]:
    async_first_json = async_first_dir / "paired-server-async.json"
    server_first_json = server_first_dir / "paired-server-async.json"
    async_first_csv = async_first_dir / "paired-server-async-summary.csv"
    server_first_csv = server_first_dir / "paired-server-async-summary.csv"

    async_first_payload = json.loads(async_first_json.read_text(encoding="utf-8"))
    server_first_payload = json.loads(server_first_json.read_text(encoding="utf-8"))
    async_first_rows = _read_csv_by_key(async_first_csv, "scenario_id")
    server_first_rows = _read_csv_by_key(server_first_csv, "scenario_id")
    scenario_ids = sorted(set(async_first_rows) & set(server_first_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for phase-order comparison")

    ratio_fields = (
        "server_to_async_output_tokens_per_second_ratio_median",
        "server_to_async_p95_first_event_ms_ratio_median",
        "server_to_async_p95_latency_ms_ratio_median",
        "server_to_async_p95_stream_tpot_ms_ratio_median",
        "server_to_async_batch_wall_ms_ratio_median",
    )
    rows = []
    for scenario_id in scenario_ids:
        async_first = async_first_rows[scenario_id]
        server_first = server_first_rows[scenario_id]
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": async_first["prompt_profile"],
            "request_count": int(async_first["request_count"]),
            "max_new_tokens": int(async_first["max_new_tokens"]),
            "pairs": int(async_first["pairs"]),
        }
        for field in ratio_fields:
            async_first_value = _float_field(async_first, field)
            server_first_value = _float_field(server_first, field)
            short_field = field.removeprefix("server_to_async_").removesuffix("_median")
            row[f"async_first_{short_field}"] = async_first_value
            row[f"server_first_{short_field}"] = server_first_value
            row[f"server_first_minus_async_first_{short_field}"] = _delta(
                server_first_value,
                async_first_value,
            )
        rows.append(row)

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-phase-order-compare",
        "async_first_dir": str(async_first_dir),
        "server_first_dir": str(server_first_dir),
        "async_first_json": str(async_first_json),
        "server_first_json": str(server_first_json),
        "async_first_csv": str(async_first_csv),
        "server_first_csv": str(server_first_csv),
        "scenario_count": len(rows),
        "async_first": {
            "phase_order": async_first_payload.get("phase_order", "async_first"),
            "paired_run_count": async_first_payload["paired_run_count"],
            "server_ready_ms": async_first_payload["server_ready_ms"],
            "async_engine_load_ms": async_first_payload["async_engine_load_ms"],
            "mean_server_to_async_throughput_ratio": async_first_payload[
                "mean_server_to_async_throughput_ratio"
            ],
            "mean_server_to_async_first_event_ratio": async_first_payload[
                "mean_server_to_async_first_event_ratio"
            ],
            "mean_server_to_async_latency_ratio": async_first_payload[
                "mean_server_to_async_latency_ratio"
            ],
            "mean_server_to_async_tpot_ratio": async_first_payload[
                "mean_server_to_async_tpot_ratio"
            ],
        },
        "server_first": {
            "phase_order": server_first_payload.get("phase_order", "server_first"),
            "paired_run_count": server_first_payload["paired_run_count"],
            "server_ready_ms": server_first_payload["server_ready_ms"],
            "async_engine_load_ms": server_first_payload["async_engine_load_ms"],
            "mean_server_to_async_throughput_ratio": server_first_payload[
                "mean_server_to_async_throughput_ratio"
            ],
            "mean_server_to_async_first_event_ratio": server_first_payload[
                "mean_server_to_async_first_event_ratio"
            ],
            "mean_server_to_async_latency_ratio": server_first_payload[
                "mean_server_to_async_latency_ratio"
            ],
            "mean_server_to_async_tpot_ratio": server_first_payload[
                "mean_server_to_async_tpot_ratio"
            ],
        },
        "rows": rows,
    }


def _aggregate_vllm_server_async_phase_order_trials(
    compare_dirs: list[Path],
) -> dict[str, Any]:
    if not compare_dirs:
        raise ValueError("compare_dirs must not be empty")

    metric_sources = {
        "throughput_ratio": "mean_server_to_async_throughput_ratio",
        "first_event_ratio": "mean_server_to_async_first_event_ratio",
        "latency_ratio": "mean_server_to_async_latency_ratio",
        "tpot_ratio": "mean_server_to_async_tpot_ratio",
    }
    trials = []
    for index, compare_dir in enumerate(compare_dirs, start=1):
        path = compare_dir / "phase-order-compare.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        trial: dict[str, Any] = {
            "trial_index": index,
            "trial_label": compare_dir.name,
            "compare_json": str(path),
            "scenario_count": payload["scenario_count"],
        }
        for metric, source_field in metric_sources.items():
            async_first_value = payload["async_first"][source_field]
            server_first_value = payload["server_first"][source_field]
            trial[f"async_first_{metric}"] = async_first_value
            trial[f"server_first_{metric}"] = server_first_value
            trial[f"server_first_minus_async_first_{metric}"] = (
                server_first_value - async_first_value
            )
        trials.append(trial)

    summary = []
    for metric in metric_sources:
        async_stats = _metric_distribution(trials, f"async_first_{metric}")
        server_stats = _metric_distribution(trials, f"server_first_{metric}")
        delta_stats = _metric_distribution(
            trials,
            f"server_first_minus_async_first_{metric}",
        )
        delta_values = [
            trial[f"server_first_minus_async_first_{metric}"]
            for trial in trials
            if trial.get(f"server_first_minus_async_first_{metric}") is not None
        ]
        bootstrap_interval = _bootstrap_mean_interval(delta_values)
        summary.append(
            {
                "metric": metric,
                "trial_count": len(trials),
                "async_first_mean": async_stats[f"async_first_{metric}_mean"],
                "async_first_min": async_stats[f"async_first_{metric}_min"],
                "async_first_max": async_stats[f"async_first_{metric}_max"],
                "async_first_cv": async_stats[f"async_first_{metric}_cv"],
                "server_first_mean": server_stats[f"server_first_{metric}_mean"],
                "server_first_min": server_stats[f"server_first_{metric}_min"],
                "server_first_max": server_stats[f"server_first_{metric}_max"],
                "server_first_cv": server_stats[f"server_first_{metric}_cv"],
                "server_first_minus_async_first_mean": delta_stats[
                    f"server_first_minus_async_first_{metric}_mean"
                ],
                "server_first_minus_async_first_min": delta_stats[
                    f"server_first_minus_async_first_{metric}_min"
                ],
                "server_first_minus_async_first_max": delta_stats[
                    f"server_first_minus_async_first_{metric}_max"
                ],
                "server_first_minus_async_first_cv": delta_stats[
                    f"server_first_minus_async_first_{metric}_cv"
                ],
                "server_first_minus_async_first_bootstrap_mean_p05": bootstrap_interval[
                    "mean_p05"
                ],
                "server_first_minus_async_first_bootstrap_mean_p50": bootstrap_interval[
                    "mean_p50"
                ],
                "server_first_minus_async_first_bootstrap_mean_p95": bootstrap_interval[
                    "mean_p95"
                ],
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-multitrial-aggregate",
        "trial_count": len(trials),
        "compare_dirs": [str(path) for path in compare_dirs],
        "trials": trials,
        "summary": summary,
    }


def _compare_vllm_server_async_workload_profiles(
    short_multitrial_dir: Path,
    long_phase_order_compare_dir: Path,
) -> dict[str, Any]:
    short_json = short_multitrial_dir / "phase-order-multitrial.json"
    short_csv = short_multitrial_dir / "phase-order-multitrial.csv"
    long_json = long_phase_order_compare_dir / "phase-order-compare.json"
    long_csv = long_phase_order_compare_dir / "phase-order-compare.csv"

    short_payload = json.loads(short_json.read_text(encoding="utf-8"))
    long_payload = json.loads(long_json.read_text(encoding="utf-8"))
    short_csv_rows = _read_csv_by_key(short_csv, "metric")
    long_csv_rows = _read_csv_by_key(long_csv, "scenario_id")
    long_profiles = sorted(
        {
            row["prompt_profile"]
            for row in long_csv_rows.values()
            if row.get("prompt_profile")
        }
    )
    long_profile = ",".join(long_profiles) if long_profiles else "long"

    metric_sources = {
        "throughput_ratio": "mean_server_to_async_throughput_ratio",
        "first_event_ratio": "mean_server_to_async_first_event_ratio",
        "latency_ratio": "mean_server_to_async_latency_ratio",
        "tpot_ratio": "mean_server_to_async_tpot_ratio",
    }
    short_summary_by_metric = {
        row["metric"]: row
        for row in short_payload.get("summary", [])
    }
    missing_metrics = sorted(set(metric_sources) - set(short_summary_by_metric))
    if missing_metrics:
        raise ValueError(
            "short multitrial aggregate is missing metrics: "
            + ", ".join(missing_metrics)
        )
    missing_csv_metrics = sorted(set(metric_sources) - set(short_csv_rows))
    if missing_csv_metrics:
        raise ValueError(
            "short multitrial csv is missing metrics: "
            + ", ".join(missing_csv_metrics)
        )

    rows = []
    for metric, source_field in metric_sources.items():
        short_row = short_summary_by_metric[metric]
        short_async_mean = short_row["async_first_mean"]
        short_server_mean = short_row["server_first_mean"]
        short_order_effect_delta_mean = short_row[
            "server_first_minus_async_first_mean"
        ]
        long_async_mean = long_payload["async_first"][source_field]
        long_server_mean = long_payload["server_first"][source_field]
        long_order_effect_delta_mean = _delta(long_server_mean, long_async_mean)

        rows.append(
            {
                "metric": metric,
                "short_profile": "short",
                "long_profile": long_profile,
                "short_trial_count": short_payload["trial_count"],
                "long_scenario_count": long_payload["scenario_count"],
                "long_paired_run_count": min(
                    long_payload["async_first"]["paired_run_count"],
                    long_payload["server_first"]["paired_run_count"],
                ),
                "short_async_first_mean": short_async_mean,
                "long_async_first_mean": long_async_mean,
                "long_minus_short_async_first_mean": _delta(
                    long_async_mean,
                    short_async_mean,
                ),
                "short_server_first_mean": short_server_mean,
                "long_server_first_mean": long_server_mean,
                "long_minus_short_server_first_mean": _delta(
                    long_server_mean,
                    short_server_mean,
                ),
                "short_order_effect_delta_mean": short_order_effect_delta_mean,
                "long_order_effect_delta_mean": long_order_effect_delta_mean,
                "long_minus_short_order_effect_delta_mean": _delta(
                    long_order_effect_delta_mean,
                    short_order_effect_delta_mean,
                ),
                "short_order_effect_bootstrap_mean_p05": short_row[
                    "server_first_minus_async_first_bootstrap_mean_p05"
                ],
                "short_order_effect_bootstrap_mean_p50": short_row[
                    "server_first_minus_async_first_bootstrap_mean_p50"
                ],
                "short_order_effect_bootstrap_mean_p95": short_row[
                    "server_first_minus_async_first_bootstrap_mean_p95"
                ],
            }
        )

    return {
        "schema_version": 1,
        "mode": "vllm-server-async-workload-compare",
        "short_profile": "short",
        "long_profile": long_profile,
        "short_multitrial_dir": str(short_multitrial_dir),
        "long_phase_order_compare_dir": str(long_phase_order_compare_dir),
        "short_json": str(short_json),
        "short_csv": str(short_csv),
        "long_json": str(long_json),
        "long_csv": str(long_csv),
        "metric_count": len(rows),
        "short_trial_count": short_payload["trial_count"],
        "long_scenario_count": long_payload["scenario_count"],
        "long_paired_run_count": min(
            long_payload["async_first"]["paired_run_count"],
            long_payload["server_first"]["paired_run_count"],
        ),
        "rows": rows,
    }


def _bootstrap_mean_interval(
    values: list[float],
    samples: int = 4096,
    seed: int = DEFAULT_VLLM_SWEEP_SEED,
) -> dict[str, float | None]:
    if not values:
        return {"mean_p05": None, "mean_p50": None, "mean_p95": None}
    if len(values) == 1:
        return {
            "mean_p05": values[0],
            "mean_p50": values[0],
            "mean_p95": values[0],
        }

    import random

    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        draw = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(draw) / len(draw))
    return {
        "mean_p05": _percentile(means, 5),
        "mean_p50": _percentile(means, 50),
        "mean_p95": _percentile(means, 95),
    }


def _make_vllm_server_async_paired_rows(
    async_runs: list[dict[str, Any]],
    server_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    async_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in async_runs
    }
    server_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in server_runs
    }
    keys = sorted(
        set(async_by_key) & set(server_by_key),
        key=lambda key: (key[0], key[1]),
    )
    if not keys:
        raise ValueError("No matching paired runs found")

    paired_rows = []
    for scenario_id, repeat_index in keys:
        async_run = async_by_key[(scenario_id, repeat_index)]
        server_run = server_by_key[(scenario_id, repeat_index)]
        async_throughput = async_run.get("aggregate_output_tokens_per_second")
        server_throughput = server_run.get("aggregate_output_tokens_per_second")
        async_first_event = async_run.get("p95_first_chunk_ms")
        server_first_event = server_run.get("p95_first_content_ms")
        async_latency = async_run.get("p95_latency_ms")
        server_latency = server_run.get("p95_latency_ms")
        async_tpot = async_run.get("p95_stream_tpot_ms")
        server_tpot = server_run.get("p95_stream_tpot_ms")
        async_batch_wall = async_run.get("batch_wall_ms")
        server_batch_wall = server_run.get("batch_wall_ms")
        paired_rows.append(
            {
                "pair_id": f"{scenario_id}_rep{repeat_index:02d}",
                "scenario_id": scenario_id,
                "prompt_profile": async_run["prompt_profile"],
                "request_count": async_run["request_count"],
                "max_new_tokens": async_run["max_new_tokens"],
                "repeat_index": repeat_index,
                "async_run_order": async_run["run_order"],
                "server_run_order": server_run["run_order"],
                "prompt_tokens_mean": async_run["prompt_tokens_mean"],
                "estimated_peak_sequence_tokens_async": async_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "estimated_peak_sequence_tokens_server": server_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "async_output_tokens_per_second": async_throughput,
                "server_output_tokens_per_second": server_throughput,
                "server_to_async_output_tokens_per_second_delta": _delta(
                    server_throughput,
                    async_throughput,
                ),
                "server_to_async_output_tokens_per_second_ratio": _ratio(
                    server_throughput,
                    async_throughput,
                ),
                "async_p95_first_event_ms": async_first_event,
                "server_p95_first_content_ms": server_first_event,
                "server_to_async_p95_first_event_ms_delta": _delta(
                    server_first_event,
                    async_first_event,
                ),
                "server_to_async_p95_first_event_ms_ratio": _ratio(
                    server_first_event,
                    async_first_event,
                ),
                "async_p95_latency_ms": async_latency,
                "server_p95_latency_ms": server_latency,
                "server_to_async_p95_latency_ms_delta": _delta(
                    server_latency,
                    async_latency,
                ),
                "server_to_async_p95_latency_ms_ratio": _ratio(
                    server_latency,
                    async_latency,
                ),
                "async_p95_stream_tpot_ms": async_tpot,
                "server_p95_stream_tpot_ms": server_tpot,
                "server_to_async_p95_stream_tpot_ms_delta": _delta(
                    server_tpot,
                    async_tpot,
                ),
                "server_to_async_p95_stream_tpot_ms_ratio": _ratio(
                    server_tpot,
                    async_tpot,
                ),
                "async_batch_wall_ms": async_batch_wall,
                "server_batch_wall_ms": server_batch_wall,
                "server_to_async_batch_wall_ms_delta": _delta(
                    server_batch_wall,
                    async_batch_wall,
                ),
                "server_to_async_batch_wall_ms_ratio": _ratio(
                    server_batch_wall,
                    async_batch_wall,
                ),
            }
        )
    return paired_rows


def _aggregate_vllm_server_async_paired_scenarios(
    paired_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in paired_runs:
        pairs_by_scenario.setdefault(row["scenario_id"], []).append(row)

    scenarios = []
    for scenario_id, pairs in sorted(pairs_by_scenario.items()):
        first = pairs[0]
        scenario: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": first["prompt_profile"],
            "request_count": first["request_count"],
            "max_new_tokens": first["max_new_tokens"],
            "pairs": len(pairs),
            "prompt_tokens_mean": first["prompt_tokens_mean"],
        }
        for field in (
            "server_to_async_output_tokens_per_second_ratio",
            "server_to_async_output_tokens_per_second_delta",
            "server_to_async_p95_first_event_ms_ratio",
            "server_to_async_p95_first_event_ms_delta",
            "server_to_async_p95_latency_ms_ratio",
            "server_to_async_p95_latency_ms_delta",
            "server_to_async_p95_stream_tpot_ms_ratio",
            "server_to_async_p95_stream_tpot_ms_delta",
            "server_to_async_batch_wall_ms_ratio",
            "server_to_async_batch_wall_ms_delta",
            "async_output_tokens_per_second",
            "server_output_tokens_per_second",
            "async_p95_first_event_ms",
            "server_p95_first_content_ms",
            "async_p95_latency_ms",
            "server_p95_latency_ms",
            "async_p95_stream_tpot_ms",
            "server_p95_stream_tpot_ms",
        ):
            scenario.update(_metric_distribution(pairs, field))
        scenarios.append(scenario)
    return scenarios


def _vllm_server_async_paired_summary_rows(
    paired_scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "pairs",
        "prompt_tokens_mean",
        "server_to_async_output_tokens_per_second_ratio_median",
        "server_to_async_output_tokens_per_second_ratio_p95",
        "server_to_async_p95_first_event_ms_ratio_median",
        "server_to_async_p95_first_event_ms_ratio_p95",
        "server_to_async_p95_latency_ms_ratio_median",
        "server_to_async_p95_latency_ms_ratio_p95",
        "server_to_async_p95_stream_tpot_ms_ratio_median",
        "server_to_async_p95_stream_tpot_ms_ratio_p95",
        "server_to_async_batch_wall_ms_ratio_median",
        "server_to_async_batch_wall_ms_ratio_p95",
        "async_output_tokens_per_second_median",
        "server_output_tokens_per_second_median",
        "async_p95_first_event_ms_median",
        "server_p95_first_content_ms_median",
        "async_p95_latency_ms_median",
        "server_p95_latency_ms_median",
        "async_p95_stream_tpot_ms_median",
        "server_p95_stream_tpot_ms_median",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in paired_scenarios
    ]


def _make_vllm_prefix_cache_paired_rows(
    cold_runs: list[dict[str, Any]],
    cache_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    cold_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in cold_runs
    }
    cache_by_key = {
        (run["scenario_id"], run["repeat_index"]): run
        for run in cache_runs
    }
    keys = sorted(
        set(cold_by_key) & set(cache_by_key),
        key=lambda key: (key[0], key[1]),
    )
    if not keys:
        raise ValueError("No matching paired prefix-cache runs found")

    paired_rows = []
    for scenario_id, repeat_index in keys:
        cold_run = cold_by_key[(scenario_id, repeat_index)]
        cache_run = cache_by_key[(scenario_id, repeat_index)]
        cold_throughput = cold_run.get("aggregate_output_tokens_per_second")
        cache_throughput = cache_run.get("aggregate_output_tokens_per_second")
        cold_first_event = cold_run.get("p95_first_chunk_ms")
        cache_first_event = cache_run.get("p95_first_chunk_ms")
        cold_latency = cold_run.get("p95_latency_ms")
        cache_latency = cache_run.get("p95_latency_ms")
        cold_tpot = cold_run.get("p95_stream_tpot_ms")
        cache_tpot = cache_run.get("p95_stream_tpot_ms")
        cold_batch_wall = cold_run.get("batch_wall_ms")
        cache_batch_wall = cache_run.get("batch_wall_ms")
        cold_prefix_hit_rate = cold_run.get("prefix_cache_hit_rate_pct")
        cache_prefix_hit_rate = cache_run.get("prefix_cache_hit_rate_pct")
        cold_gpu_kv_usage = cold_run.get("gpu_kv_cache_usage_pct")
        cache_gpu_kv_usage = cache_run.get("gpu_kv_cache_usage_pct")
        paired_rows.append(
            {
                "pair_id": f"{scenario_id}_rep{repeat_index:02d}",
                "scenario_id": scenario_id,
                "prompt_profile": cold_run["prompt_profile"],
                "request_count": cold_run["request_count"],
                "max_new_tokens": cold_run["max_new_tokens"],
                "repeat_index": repeat_index,
                "cold_run_order": cold_run["run_order"],
                "cache_run_order": cache_run["run_order"],
                "prompt_tokens_mean": cold_run["prompt_tokens_mean"],
                "estimated_peak_sequence_tokens_cold": cold_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "estimated_peak_sequence_tokens_cache": cache_run.get(
                    "estimated_peak_sequence_tokens"
                ),
                "cold_output_tokens_per_second": cold_throughput,
                "cache_output_tokens_per_second": cache_throughput,
                "cache_to_cold_output_tokens_per_second_delta": _delta(
                    cache_throughput,
                    cold_throughput,
                ),
                "cache_to_cold_output_tokens_per_second_ratio": _ratio(
                    cache_throughput,
                    cold_throughput,
                ),
                "cold_p95_first_event_ms": cold_first_event,
                "cache_p95_first_event_ms": cache_first_event,
                "cache_to_cold_p95_first_event_ms_delta": _delta(
                    cache_first_event,
                    cold_first_event,
                ),
                "cache_to_cold_p95_first_event_ms_ratio": _ratio(
                    cache_first_event,
                    cold_first_event,
                ),
                "cold_p95_latency_ms": cold_latency,
                "cache_p95_latency_ms": cache_latency,
                "cache_to_cold_p95_latency_ms_delta": _delta(
                    cache_latency,
                    cold_latency,
                ),
                "cache_to_cold_p95_latency_ms_ratio": _ratio(
                    cache_latency,
                    cold_latency,
                ),
                "cold_p95_stream_tpot_ms": cold_tpot,
                "cache_p95_stream_tpot_ms": cache_tpot,
                "cache_to_cold_p95_stream_tpot_ms_delta": _delta(
                    cache_tpot,
                    cold_tpot,
                ),
                "cache_to_cold_p95_stream_tpot_ms_ratio": _ratio(
                    cache_tpot,
                    cold_tpot,
                ),
                "cold_batch_wall_ms": cold_batch_wall,
                "cache_batch_wall_ms": cache_batch_wall,
                "cache_to_cold_batch_wall_ms_delta": _delta(
                    cache_batch_wall,
                    cold_batch_wall,
                ),
                "cache_to_cold_batch_wall_ms_ratio": _ratio(
                    cache_batch_wall,
                    cold_batch_wall,
                ),
                "cold_prefix_cache_hit_rate_pct": cold_prefix_hit_rate,
                "cache_prefix_cache_hit_rate_pct": cache_prefix_hit_rate,
                "cache_to_cold_prefix_cache_hit_rate_pct_delta": _delta(
                    cache_prefix_hit_rate,
                    cold_prefix_hit_rate,
                ),
                "cold_gpu_kv_cache_usage_pct": cold_gpu_kv_usage,
                "cache_gpu_kv_cache_usage_pct": cache_gpu_kv_usage,
                "cache_to_cold_gpu_kv_cache_usage_pct_delta": _delta(
                    cache_gpu_kv_usage,
                    cold_gpu_kv_usage,
                ),
            }
        )
    return paired_rows


def _aggregate_vllm_prefix_cache_paired_scenarios(
    paired_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pairs_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in paired_runs:
        pairs_by_scenario.setdefault(row["scenario_id"], []).append(row)

    scenarios = []
    for scenario_id, pairs in sorted(pairs_by_scenario.items()):
        first = pairs[0]
        scenario: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": first["prompt_profile"],
            "request_count": first["request_count"],
            "max_new_tokens": first["max_new_tokens"],
            "pairs": len(pairs),
            "prompt_tokens_mean": first["prompt_tokens_mean"],
        }
        for field in (
            "cache_to_cold_output_tokens_per_second_ratio",
            "cache_to_cold_output_tokens_per_second_delta",
            "cache_to_cold_p95_first_event_ms_ratio",
            "cache_to_cold_p95_first_event_ms_delta",
            "cache_to_cold_p95_latency_ms_ratio",
            "cache_to_cold_p95_latency_ms_delta",
            "cache_to_cold_p95_stream_tpot_ms_ratio",
            "cache_to_cold_p95_stream_tpot_ms_delta",
            "cache_to_cold_batch_wall_ms_ratio",
            "cache_to_cold_batch_wall_ms_delta",
            "cold_output_tokens_per_second",
            "cache_output_tokens_per_second",
            "cold_p95_first_event_ms",
            "cache_p95_first_event_ms",
            "cold_p95_latency_ms",
            "cache_p95_latency_ms",
            "cold_p95_stream_tpot_ms",
            "cache_p95_stream_tpot_ms",
            "cold_prefix_cache_hit_rate_pct",
            "cache_prefix_cache_hit_rate_pct",
            "cache_to_cold_prefix_cache_hit_rate_pct_delta",
            "cold_gpu_kv_cache_usage_pct",
            "cache_gpu_kv_cache_usage_pct",
            "cache_to_cold_gpu_kv_cache_usage_pct_delta",
        ):
            scenario.update(_metric_distribution(pairs, field))
        scenarios.append(scenario)
    return scenarios


def _vllm_prefix_cache_paired_summary_rows(
    paired_scenarios: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    row_fields = (
        "scenario_id",
        "prompt_profile",
        "request_count",
        "max_new_tokens",
        "pairs",
        "prompt_tokens_mean",
        "cache_to_cold_output_tokens_per_second_ratio_median",
        "cache_to_cold_output_tokens_per_second_ratio_p95",
        "cache_to_cold_p95_first_event_ms_ratio_median",
        "cache_to_cold_p95_first_event_ms_ratio_p95",
        "cache_to_cold_p95_latency_ms_ratio_median",
        "cache_to_cold_p95_latency_ms_ratio_p95",
        "cache_to_cold_p95_stream_tpot_ms_ratio_median",
        "cache_to_cold_p95_stream_tpot_ms_ratio_p95",
        "cache_to_cold_batch_wall_ms_ratio_median",
        "cache_to_cold_batch_wall_ms_ratio_p95",
        "cold_output_tokens_per_second_median",
        "cache_output_tokens_per_second_median",
        "cold_p95_first_event_ms_median",
        "cache_p95_first_event_ms_median",
        "cold_p95_latency_ms_median",
        "cache_p95_latency_ms_median",
        "cold_p95_stream_tpot_ms_median",
        "cache_p95_stream_tpot_ms_median",
        "cold_prefix_cache_hit_rate_pct_median",
        "cache_prefix_cache_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_hit_rate_pct_delta_median",
        "cold_gpu_kv_cache_usage_pct_median",
        "cache_gpu_kv_cache_usage_pct_median",
        "cache_to_cold_gpu_kv_cache_usage_pct_delta_median",
    )
    return [
        {field: scenario.get(field) for field in row_fields}
        for scenario in paired_scenarios
    ]


def _vllm_prefix_cache_isolated_profile_control_rows(
    paired_scenarios: list[dict[str, Any]],
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> list[dict[str, Any]]:
    by_profile_and_shape = {
        (
            scenario["prompt_profile"],
            int(scenario["request_count"]),
            int(scenario["max_new_tokens"]),
        ): scenario
        for scenario in paired_scenarios
    }
    shared_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == shared_profile
    }
    control_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == control_profile
    }
    comparison_keys = sorted(shared_keys & control_keys)
    if not comparison_keys:
        return []

    metric_fields = (
        "cache_to_cold_output_tokens_per_second_ratio_median",
        "cache_to_cold_p95_first_event_ms_ratio_median",
        "cache_to_cold_p95_latency_ms_ratio_median",
        "cache_to_cold_p95_stream_tpot_ms_ratio_median",
        "cache_to_cold_batch_wall_ms_ratio_median",
        "cold_prefix_cache_hit_rate_pct_median",
        "cache_prefix_cache_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_hit_rate_pct_delta_median",
        "cold_gpu_kv_cache_usage_pct_median",
        "cache_gpu_kv_cache_usage_pct_median",
        "cache_to_cold_gpu_kv_cache_usage_pct_delta_median",
    )
    rows = []
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_prompt_tokens = shared.get("prompt_tokens_mean")
        control_prompt_tokens = control.get("prompt_tokens_mean")
        row: dict[str, Any] = {
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "shared_profile": shared_profile,
            "control_profile": control_profile,
            "shared_scenario_id": shared["scenario_id"],
            "control_scenario_id": control["scenario_id"],
            "shared_prompt_tokens_mean": shared_prompt_tokens,
            "control_prompt_tokens_mean": control_prompt_tokens,
            "shared_to_control_prompt_tokens_mean_ratio": _ratio(
                shared_prompt_tokens,
                control_prompt_tokens,
            ),
        }
        for field in metric_fields:
            shared_value = shared.get(field)
            control_value = control.get(field)
            row[f"shared_{field}"] = shared_value
            row[f"control_{field}"] = control_value
            row[f"shared_minus_control_{field}"] = _delta(
                shared_value,
                control_value,
            )
        rows.append(row)
    return rows


def _vllm_prefix_cache_isolated_profile_control_summary(
    profile_control_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if not profile_control_rows:
        return {}
    summary_fields = [
        field
        for field in profile_control_rows[0]
        if field.startswith("shared_minus_control_")
    ]
    summary = {
        f"mean_{field}": _mean_present(row[field] for row in profile_control_rows)
        for field in summary_fields
    }
    summary["mean_shared_to_control_prompt_tokens_mean_ratio"] = _mean_present(
        row["shared_to_control_prompt_tokens_mean_ratio"]
        for row in profile_control_rows
    )
    return summary


def _latest_cache_metric_field(
    metrics: list[dict[str, Any]] | None,
    field: str,
) -> float | None:
    for row in reversed(metrics or []):
        value = row.get(field)
        if value is not None:
            return float(value)
    return None


def _estimated_window_rate_pct(
    *,
    warmup_rate_pct: float | None,
    after_rate_pct: float | None,
    warmup_tokens: float | None,
    measured_tokens: float | None,
) -> float | None:
    if (
        warmup_rate_pct is None
        or after_rate_pct is None
        or warmup_tokens is None
        or measured_tokens is None
        or measured_tokens <= 0
    ):
        return None
    total_tokens = warmup_tokens + measured_tokens
    estimated = (
        (after_rate_pct / 100.0 * total_tokens)
        - (warmup_rate_pct / 100.0 * warmup_tokens)
    ) / measured_tokens * 100.0
    return max(0.0, min(100.0, estimated))


def _summarize_vllm_prefix_cache_isolated_window(
    isolated_metrics_dir: Path,
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> dict[str, Any]:
    source_json = isolated_metrics_dir / "prefix-cache-isolated-metrics.json"
    source_payload = json.loads(source_json.read_text(encoding="utf-8"))
    paired_runs = source_payload["paired_runs"]
    remote_calls = {
        int(row["isolation_call_index"]): row
        for row in source_payload["remote_call_summaries"]
    }
    window_rows = []
    for paired in paired_runs:
        call = remote_calls[int(paired["isolation_call_index"])]
        warmup_summaries = call.get("cache_warmup", {}).get("summaries", [])
        warmup_prompt_tokens = sum(
            float(row.get("total_prompt_tokens") or 0.0)
            for row in warmup_summaries
        )
        measured_prompt_tokens = (
            float(paired["prompt_tokens_mean"]) * int(paired["request_count"])
        )
        cache_warmup_hit = _latest_cache_metric_field(
            call.get("cache_warmup_cache_metrics"),
            "prefix_cache_hit_rate_pct",
        )
        cache_after_hit = paired.get("cache_prefix_cache_hit_rate_pct")
        cold_warmup_hit = _latest_cache_metric_field(
            call.get("cold_warmup_cache_metrics"),
            "prefix_cache_hit_rate_pct",
        )
        cold_after_hit = paired.get("cold_prefix_cache_hit_rate_pct")
        cache_estimated = _estimated_window_rate_pct(
            warmup_rate_pct=cache_warmup_hit,
            after_rate_pct=cache_after_hit,
            warmup_tokens=warmup_prompt_tokens,
            measured_tokens=measured_prompt_tokens,
        )
        cold_estimated = _estimated_window_rate_pct(
            warmup_rate_pct=cold_warmup_hit,
            after_rate_pct=cold_after_hit,
            warmup_tokens=warmup_prompt_tokens,
            measured_tokens=measured_prompt_tokens,
        )
        window_rows.append(
            {
                "scenario_id": paired["scenario_id"],
                "prompt_profile": paired["prompt_profile"],
                "request_count": int(paired["request_count"]),
                "max_new_tokens": int(paired["max_new_tokens"]),
                "repeat_index": int(paired["repeat_index"]),
                "isolation_call_index": int(paired["isolation_call_index"]),
                "warmup_prompt_profile": call.get("warmup_prompt_profile"),
                "warmup_prompt_tokens": warmup_prompt_tokens,
                "measured_prompt_tokens": measured_prompt_tokens,
                "cache_warmup_prefix_cache_hit_rate_pct": cache_warmup_hit,
                "cache_after_prefix_cache_hit_rate_pct": cache_after_hit,
                "cache_estimated_measured_prefix_cache_hit_rate_pct": cache_estimated,
                "cold_warmup_prefix_cache_hit_rate_pct": cold_warmup_hit,
                "cold_after_prefix_cache_hit_rate_pct": cold_after_hit,
                "cold_estimated_measured_prefix_cache_hit_rate_pct": cold_estimated,
                "cache_to_cold_estimated_measured_prefix_cache_hit_rate_pct_delta": _delta(
                    cache_estimated,
                    cold_estimated,
                ),
                "cache_to_cold_output_tokens_per_second_ratio": paired[
                    "cache_to_cold_output_tokens_per_second_ratio"
                ],
                "cache_to_cold_p95_latency_ms_ratio": paired[
                    "cache_to_cold_p95_latency_ms_ratio"
                ],
            }
        )

    def add_stats(row: dict[str, Any], prefix: str, values: list[float]) -> None:
        row[f"{prefix}_mean"] = sum(values) / len(values) if values else None
        row[f"{prefix}_min"] = min(values) if values else None
        row[f"{prefix}_max"] = max(values) if values else None
        row[f"{prefix}_population_stdev"] = (
            statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
        )

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in window_rows:
        key = (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        )
        grouped.setdefault(key, []).append(row)

    scenario_rows = []
    for key, rows in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[0][2], item[0][1]),
    ):
        prompt_profile, request_count, max_new_tokens = key
        first = rows[0]
        scenario: dict[str, Any] = {
            "scenario_id": first["scenario_id"],
            "prompt_profile": prompt_profile,
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "runs": len(rows),
            "warmup_prompt_profile": first["warmup_prompt_profile"],
            "warmup_prompt_tokens_mean": _mean_present(
                row["warmup_prompt_tokens"] for row in rows
            ),
            "measured_prompt_tokens_mean": _mean_present(
                row["measured_prompt_tokens"] for row in rows
            ),
        }
        for field in (
            "cache_warmup_prefix_cache_hit_rate_pct",
            "cache_after_prefix_cache_hit_rate_pct",
            "cache_estimated_measured_prefix_cache_hit_rate_pct",
            "cold_estimated_measured_prefix_cache_hit_rate_pct",
            "cache_to_cold_estimated_measured_prefix_cache_hit_rate_pct_delta",
            "cache_to_cold_output_tokens_per_second_ratio",
            "cache_to_cold_p95_latency_ms_ratio",
        ):
            values = [
                float(row[field])
                for row in rows
                if row.get(field) is not None
            ]
            add_stats(scenario, field, values)
        scenario_rows.append(scenario)

    by_profile_and_shape = {
        (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        ): row
        for row in scenario_rows
    }
    comparison_keys = sorted(
        {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == shared_profile
        }
        & {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == control_profile
        }
    )
    profile_control_rows = []
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_estimated = shared[
            "cache_estimated_measured_prefix_cache_hit_rate_pct_mean"
        ]
        control_estimated = control[
            "cache_estimated_measured_prefix_cache_hit_rate_pct_mean"
        ]
        shared_after = shared["cache_after_prefix_cache_hit_rate_pct_mean"]
        control_after = control["cache_after_prefix_cache_hit_rate_pct_mean"]
        shared_throughput = shared[
            "cache_to_cold_output_tokens_per_second_ratio_mean"
        ]
        control_throughput = control[
            "cache_to_cold_output_tokens_per_second_ratio_mean"
        ]
        shared_latency = shared["cache_to_cold_p95_latency_ms_ratio_mean"]
        control_latency = control["cache_to_cold_p95_latency_ms_ratio_mean"]
        profile_control_rows.append(
            {
                "request_count": request_count,
                "max_new_tokens": max_new_tokens,
                "shared_profile": shared_profile,
                "control_profile": control_profile,
                "shared_estimated_measured_cache_hit_rate_pct": shared_estimated,
                "control_estimated_measured_cache_hit_rate_pct": control_estimated,
                "shared_minus_control_estimated_measured_cache_hit_rate_pct": _delta(
                    shared_estimated,
                    control_estimated,
                ),
                "shared_after_cache_hit_rate_pct": shared_after,
                "control_after_cache_hit_rate_pct": control_after,
                "shared_minus_control_after_cache_hit_rate_pct": _delta(
                    shared_after,
                    control_after,
                ),
                "shared_cache_to_cold_throughput_ratio_mean": shared_throughput,
                "control_cache_to_cold_throughput_ratio_mean": control_throughput,
                "shared_minus_control_cache_to_cold_throughput_ratio_mean": _delta(
                    shared_throughput,
                    control_throughput,
                ),
                "shared_cache_to_cold_p95_latency_ratio_mean": shared_latency,
                "control_cache_to_cold_p95_latency_ratio_mean": control_latency,
                "shared_minus_control_cache_to_cold_p95_latency_ratio_mean": _delta(
                    shared_latency,
                    control_latency,
                ),
            }
        )

    estimated_stdevs = [
        row["cache_estimated_measured_prefix_cache_hit_rate_pct_population_stdev"]
        for row in scenario_rows
        if row["cache_estimated_measured_prefix_cache_hit_rate_pct_population_stdev"]
        is not None
    ]
    summary = {
        "mean_shared_minus_control_estimated_measured_cache_hit_rate_pct": _mean_present(
            row["shared_minus_control_estimated_measured_cache_hit_rate_pct"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_after_cache_hit_rate_pct": _mean_present(
            row["shared_minus_control_after_cache_hit_rate_pct"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_throughput_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_throughput_ratio_mean"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_p95_latency_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_p95_latency_ratio_mean"]
            for row in profile_control_rows
        ),
        "max_estimated_measured_cache_hit_rate_population_stdev": (
            max(estimated_stdevs) if estimated_stdevs else None
        ),
    }
    markdown = _format_vllm_prefix_cache_isolated_window_markdown(
        source_dir=isolated_metrics_dir,
        source_mode=source_payload.get("mode"),
        source_warmup_runs=source_payload.get("warmup_runs"),
        source_warmup_prompt_profile=source_payload.get("warmup_prompt_profile"),
        scenario_rows=scenario_rows,
        profile_control_rows=profile_control_rows,
        summary=summary,
    )
    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-isolated-window-summary",
        "source_dir": str(isolated_metrics_dir),
        "source_json": str(source_json),
        "source_mode": source_payload.get("mode"),
        "source_warmup_runs": source_payload.get("warmup_runs"),
        "source_warmup_prompt_profile": source_payload.get("warmup_prompt_profile"),
        "source_scenario_count": source_payload["scenario_count"],
        "source_paired_run_count": source_payload["paired_run_count"],
        "source_remote_call_count": source_payload["remote_call_count"],
        "estimate_method": (
            "Token-weighted before/after rate estimate: "
            "after_rate * (warmup_prompt_tokens + measured_prompt_tokens) "
            "minus warmup_rate * warmup_prompt_tokens, divided by measured_prompt_tokens."
        ),
        "scenario_count": len(scenario_rows),
        "profile_control_row_count": len(profile_control_rows),
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "summary": summary,
        "window_rows": window_rows,
        "scenario_rows": scenario_rows,
        "profile_control_rows": profile_control_rows,
        "markdown": markdown,
    }


def _format_vllm_prefix_cache_isolated_window_markdown(
    source_dir: Path,
    source_mode: Any,
    source_warmup_runs: Any,
    source_warmup_prompt_profile: Any,
    scenario_rows: list[dict[str, Any]],
    profile_control_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    def fmt(value: Any, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.3f}{suffix}"

    lines = [
        "# Prefix-Cache Measured-Window Estimate",
        "",
        f"Source: `{source_dir}`",
        f"Source mode: `{source_mode}`",
        f"Warmup runs: `{source_warmup_runs}`",
        f"Warmup prompt profile: `{source_warmup_prompt_profile}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            "| Mean shared-minus-control estimated measured cache hit rate | "
            f"{fmt(summary['mean_shared_minus_control_estimated_measured_cache_hit_rate_pct'], ' pp')} |"
        ),
        (
            "| Mean shared-minus-control logged after-window cache hit rate | "
            f"{fmt(summary['mean_shared_minus_control_after_cache_hit_rate_pct'], ' pp')} |"
        ),
        "",
        "## Scenario Estimates",
        "",
        "| Profile | Requests | Runs | Warmup hit | Logged after | Estimated measured |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario_rows:
        lines.append(
            "| "
            f"`{row['prompt_profile']}` | "
            f"{row['request_count']} | "
            f"{row['runs']} | "
            f"{fmt(row['cache_warmup_prefix_cache_hit_rate_pct_mean'], '%')} | "
            f"{fmt(row['cache_after_prefix_cache_hit_rate_pct_mean'], '%')} | "
            f"{fmt(row['cache_estimated_measured_prefix_cache_hit_rate_pct_mean'], '%')} |"
        )
    lines.extend(
        [
            "",
            "## Shared Vs Control",
            "",
            "| Requests | Estimated Shared | Estimated Control | Delta | Throughput Delta | p95 Latency Delta |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in profile_control_rows:
        lines.append(
            "| "
            f"{row['request_count']} | "
            f"{fmt(row['shared_estimated_measured_cache_hit_rate_pct'], '%')} | "
            f"{fmt(row['control_estimated_measured_cache_hit_rate_pct'], '%')} | "
            f"{fmt(row['shared_minus_control_estimated_measured_cache_hit_rate_pct'], ' pp')} | "
            f"{fmt(row['shared_minus_control_cache_to_cold_throughput_ratio_mean'])} | "
            f"{fmt(row['shared_minus_control_cache_to_cold_p95_latency_ratio_mean'])} |"
        )
    lines.extend(
        [
            "",
            "Estimate method: token-weighted before/after rate delta. This is an approximation until direct vLLM hit/miss counters are captured.",
            "",
        ]
    )
    return "\n".join(lines)


def _summarize_vllm_prefix_cache_isolated_stability(
    isolated_metrics_dir: Path,
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> dict[str, Any]:
    source_json = isolated_metrics_dir / "prefix-cache-isolated-metrics.json"
    source_runs_csv = isolated_metrics_dir / "prefix-cache-isolated-metrics-runs.csv"
    source_payload = json.loads(source_json.read_text(encoding="utf-8"))
    with source_runs_csv.open("r", encoding="utf-8", newline="") as handle:
        run_rows = list(csv.DictReader(handle))
    if not run_rows:
        raise ValueError("No isolated metric run rows found")

    def present_values(rows: list[dict[str, Any]], field: str) -> list[float]:
        values = []
        for row in rows:
            value = _float_field(row, field)
            if value is not None:
                values.append(value)
        return values

    def add_stats(row: dict[str, Any], prefix: str, values: list[float]) -> None:
        row[f"{prefix}_mean"] = sum(values) / len(values) if values else None
        row[f"{prefix}_min"] = min(values) if values else None
        row[f"{prefix}_max"] = max(values) if values else None
        row[f"{prefix}_population_stdev"] = (
            statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None
        )

    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for row in run_rows:
        key = (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        )
        grouped.setdefault(key, []).append(row)

    scenario_rows = []
    for key, rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][2], item[0][1])):
        prompt_profile, request_count, max_new_tokens = key
        first = rows[0]
        scenario: dict[str, Any] = {
            "scenario_id": first["scenario_id"],
            "prompt_profile": prompt_profile,
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "runs": len(rows),
            "prompt_tokens_mean": _float_field(first, "prompt_tokens_mean"),
        }
        for field in (
            "cold_prefix_cache_hit_rate_pct",
            "cache_prefix_cache_hit_rate_pct",
            "cache_to_cold_prefix_cache_hit_rate_pct_delta",
            "cache_to_cold_output_tokens_per_second_ratio",
            "cache_to_cold_p95_latency_ms_ratio",
            "cache_to_cold_p95_stream_tpot_ms_ratio",
        ):
            add_stats(scenario, field, present_values(rows, field))
        scenario_rows.append(scenario)

    by_profile_and_shape = {
        (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        ): row
        for row in scenario_rows
    }
    comparison_keys = sorted(
        {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == shared_profile
        }
        & {
            (request_count, max_new_tokens)
            for profile, request_count, max_new_tokens in by_profile_and_shape
            if profile == control_profile
        }
    )
    profile_control_rows = []
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_hit = shared["cache_prefix_cache_hit_rate_pct_mean"]
        control_hit = control["cache_prefix_cache_hit_rate_pct_mean"]
        shared_throughput = shared["cache_to_cold_output_tokens_per_second_ratio_mean"]
        control_throughput = control["cache_to_cold_output_tokens_per_second_ratio_mean"]
        shared_latency = shared["cache_to_cold_p95_latency_ms_ratio_mean"]
        control_latency = control["cache_to_cold_p95_latency_ms_ratio_mean"]
        profile_control_rows.append(
            {
                "request_count": request_count,
                "max_new_tokens": max_new_tokens,
                "shared_profile": shared_profile,
                "control_profile": control_profile,
                "shared_cache_hit_rate_pct_mean": shared_hit,
                "control_cache_hit_rate_pct_mean": control_hit,
                "shared_minus_control_cache_hit_rate_pct": _delta(
                    shared_hit,
                    control_hit,
                ),
                "shared_cache_hit_rate_pct_population_stdev": shared[
                    "cache_prefix_cache_hit_rate_pct_population_stdev"
                ],
                "control_cache_hit_rate_pct_population_stdev": control[
                    "cache_prefix_cache_hit_rate_pct_population_stdev"
                ],
                "shared_cache_to_cold_throughput_ratio_mean": shared_throughput,
                "control_cache_to_cold_throughput_ratio_mean": control_throughput,
                "shared_minus_control_cache_to_cold_throughput_ratio_mean": _delta(
                    shared_throughput,
                    control_throughput,
                ),
                "shared_cache_to_cold_p95_latency_ratio_mean": shared_latency,
                "control_cache_to_cold_p95_latency_ratio_mean": control_latency,
                "shared_minus_control_cache_to_cold_p95_latency_ratio_mean": _delta(
                    shared_latency,
                    control_latency,
                ),
            }
        )

    cache_stdevs = [
        row["cache_prefix_cache_hit_rate_pct_population_stdev"]
        for row in scenario_rows
        if row["cache_prefix_cache_hit_rate_pct_population_stdev"] is not None
    ]
    summary = {
        "mean_shared_minus_control_cache_hit_rate_pct": _mean_present(
            row["shared_minus_control_cache_hit_rate_pct"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_throughput_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_throughput_ratio_mean"]
            for row in profile_control_rows
        ),
        "mean_shared_minus_control_cache_to_cold_p95_latency_ratio": _mean_present(
            row["shared_minus_control_cache_to_cold_p95_latency_ratio_mean"]
            for row in profile_control_rows
        ),
        "max_cache_hit_rate_population_stdev": max(cache_stdevs) if cache_stdevs else None,
    }
    source_mode = source_payload.get("mode")
    source_warmup_runs = source_payload.get("warmup_runs")
    source_warmup_prompt_profile = source_payload.get("warmup_prompt_profile")
    source_repeats = source_payload.get("repeats")
    source_phase_order = source_payload.get("phase_order")
    markdown = _format_vllm_prefix_cache_isolated_stability_markdown(
        source_dir=isolated_metrics_dir,
        source_mode=source_mode,
        source_warmup_runs=source_warmup_runs,
        source_warmup_prompt_profile=source_warmup_prompt_profile,
        source_repeats=source_repeats,
        source_phase_order=source_phase_order,
        scenario_rows=scenario_rows,
        profile_control_rows=profile_control_rows,
        summary=summary,
    )
    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-isolated-stability-summary",
        "source_dir": str(isolated_metrics_dir),
        "source_json": str(source_json),
        "source_runs_csv": str(source_runs_csv),
        "source_scenario_count": source_payload["scenario_count"],
        "source_paired_run_count": source_payload["paired_run_count"],
        "source_remote_call_count": source_payload["remote_call_count"],
        "source_mode": source_mode,
        "source_warmup_runs": source_warmup_runs,
        "source_warmup_prompt_profile": source_warmup_prompt_profile,
        "source_repeats": source_repeats,
        "source_phase_order": source_phase_order,
        "scenario_count": len(scenario_rows),
        "profile_control_row_count": len(profile_control_rows),
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "summary": summary,
        "scenario_rows": scenario_rows,
        "profile_control_rows": profile_control_rows,
        "markdown": markdown,
    }


def _format_vllm_prefix_cache_isolated_stability_markdown(
    source_dir: Path,
    source_mode: Any,
    source_warmup_runs: Any,
    source_warmup_prompt_profile: Any,
    source_repeats: Any,
    source_phase_order: Any,
    scenario_rows: list[dict[str, Any]],
    profile_control_rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    lines = [
        "# Prefix-Cache Isolated Stability Summary",
        "",
        f"Source: `{source_dir}`",
        f"Source mode: `{source_mode}`",
        f"Warmup runs: `{source_warmup_runs}`",
        f"Warmup prompt profile: `{source_warmup_prompt_profile}`",
        f"Repeats: `{source_repeats}`",
        f"Phase order: `{source_phase_order}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        (
            "| Mean shared-minus-control cache hit rate | "
            f"{summary['mean_shared_minus_control_cache_hit_rate_pct']:.3f} pp |"
        ),
        (
            "| Max cache-hit population stdev | "
            f"{summary['max_cache_hit_rate_population_stdev']:.3f} |"
        ),
        "",
        "## Scenario Stability",
        "",
        "| Profile | Requests | Runs | Cache Hit Mean | Min | Max | Stdev |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in scenario_rows:
        lines.append(
            "| "
            f"`{row['prompt_profile']}` | "
            f"{row['request_count']} | "
            f"{row['runs']} | "
            f"{row['cache_prefix_cache_hit_rate_pct_mean']:.3f}% | "
            f"{row['cache_prefix_cache_hit_rate_pct_min']:.3f}% | "
            f"{row['cache_prefix_cache_hit_rate_pct_max']:.3f}% | "
            f"{row['cache_prefix_cache_hit_rate_pct_population_stdev']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Shared Vs Control",
            "",
            (
                "| Requests | Shared Hit Mean | Control Hit Mean | "
                "Delta | Throughput Delta | p95 Latency Delta |"
            ),
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in profile_control_rows:
        lines.append(
            "| "
            f"{row['request_count']} | "
            f"{row['shared_cache_hit_rate_pct_mean']:.3f}% | "
            f"{row['control_cache_hit_rate_pct_mean']:.3f}% | "
            f"{row['shared_minus_control_cache_hit_rate_pct']:.3f} pp | "
            f"{row['shared_minus_control_cache_to_cold_throughput_ratio_mean']:.3f} | "
            f"{row['shared_minus_control_cache_to_cold_p95_latency_ratio_mean']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _compare_vllm_prefix_cache_phase_orders(
    cold_first_dir: Path,
    cache_first_dir: Path,
) -> dict[str, Any]:
    cold_first_json = cold_first_dir / "paired-prefix-cache.json"
    cache_first_json = cache_first_dir / "paired-prefix-cache.json"
    cold_first_csv = cold_first_dir / "paired-prefix-cache-summary.csv"
    cache_first_csv = cache_first_dir / "paired-prefix-cache-summary.csv"

    cold_first_payload = json.loads(cold_first_json.read_text(encoding="utf-8"))
    cache_first_payload = json.loads(cache_first_json.read_text(encoding="utf-8"))
    cold_first_rows = _read_csv_by_key(cold_first_csv, "scenario_id")
    cache_first_rows = _read_csv_by_key(cache_first_csv, "scenario_id")
    scenario_ids = sorted(set(cold_first_rows) & set(cache_first_rows))
    if not scenario_ids:
        raise ValueError("No matching scenario_id values found for prefix-cache phase comparison")

    ratio_fields = (
        "cache_to_cold_output_tokens_per_second_ratio_median",
        "cache_to_cold_p95_first_event_ms_ratio_median",
        "cache_to_cold_p95_latency_ms_ratio_median",
        "cache_to_cold_p95_stream_tpot_ms_ratio_median",
        "cache_to_cold_batch_wall_ms_ratio_median",
        "cold_prefix_cache_hit_rate_pct_median",
        "cache_prefix_cache_hit_rate_pct_median",
        "cache_to_cold_prefix_cache_hit_rate_pct_delta_median",
        "cold_gpu_kv_cache_usage_pct_median",
        "cache_gpu_kv_cache_usage_pct_median",
        "cache_to_cold_gpu_kv_cache_usage_pct_delta_median",
    )
    rows = []
    for scenario_id in scenario_ids:
        cold_first = cold_first_rows[scenario_id]
        cache_first = cache_first_rows[scenario_id]
        row: dict[str, Any] = {
            "scenario_id": scenario_id,
            "prompt_profile": cold_first["prompt_profile"],
            "request_count": int(cold_first["request_count"]),
            "max_new_tokens": int(cold_first["max_new_tokens"]),
            "pairs": int(cold_first["pairs"]),
            "prompt_tokens_mean": _float_field(cold_first, "prompt_tokens_mean"),
        }
        for field in ratio_fields:
            cold_first_value = _float_field(cold_first, field)
            cache_first_value = _float_field(cache_first, field)
            short_field = field.removesuffix("_median")
            if short_field.startswith("cache_to_cold_"):
                short_field = short_field.removeprefix("cache_to_cold_")
            row[f"cold_first_{short_field}"] = cold_first_value
            row[f"cache_first_{short_field}"] = cache_first_value
            row[f"cache_first_minus_cold_first_{short_field}"] = _delta(
                cache_first_value,
                cold_first_value,
            )
        rows.append(row)

    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-phase-order-compare",
        "cold_first_dir": str(cold_first_dir),
        "cache_first_dir": str(cache_first_dir),
        "cold_first_json": str(cold_first_json),
        "cache_first_json": str(cache_first_json),
        "cold_first_csv": str(cold_first_csv),
        "cache_first_csv": str(cache_first_csv),
        "scenario_count": len(rows),
        "cold_first": {
            "phase_order": cold_first_payload.get("phase_order", "cold_first"),
            "paired_run_count": cold_first_payload["paired_run_count"],
            "cold_engine_load_ms": cold_first_payload["cold_engine_load_ms"],
            "cache_engine_load_ms": cold_first_payload["cache_engine_load_ms"],
            "mean_cache_to_cold_throughput_ratio": cold_first_payload[
                "mean_cache_to_cold_throughput_ratio"
            ],
            "mean_cache_to_cold_first_event_ratio": cold_first_payload[
                "mean_cache_to_cold_first_event_ratio"
            ],
            "mean_cache_to_cold_latency_ratio": cold_first_payload[
                "mean_cache_to_cold_latency_ratio"
            ],
            "mean_cache_to_cold_tpot_ratio": cold_first_payload[
                "mean_cache_to_cold_tpot_ratio"
            ],
        },
        "cache_first": {
            "phase_order": cache_first_payload.get("phase_order", "cache_first"),
            "paired_run_count": cache_first_payload["paired_run_count"],
            "cold_engine_load_ms": cache_first_payload["cold_engine_load_ms"],
            "cache_engine_load_ms": cache_first_payload["cache_engine_load_ms"],
            "mean_cache_to_cold_throughput_ratio": cache_first_payload[
                "mean_cache_to_cold_throughput_ratio"
            ],
            "mean_cache_to_cold_first_event_ratio": cache_first_payload[
                "mean_cache_to_cold_first_event_ratio"
            ],
            "mean_cache_to_cold_latency_ratio": cache_first_payload[
                "mean_cache_to_cold_latency_ratio"
            ],
            "mean_cache_to_cold_tpot_ratio": cache_first_payload[
                "mean_cache_to_cold_tpot_ratio"
            ],
        },
        "rows": rows,
    }


def _compare_vllm_prefix_cache_profile_controls(
    phase_order_compare_dir: Path,
    shared_profile: str = "shared_prefix_long",
    control_profile: str = "matched_unique_prefix",
) -> dict[str, Any]:
    phase_json = phase_order_compare_dir / "prefix-cache-phase-order-compare.json"
    phase_csv = phase_order_compare_dir / "prefix-cache-phase-order-compare.csv"
    phase_payload = json.loads(phase_json.read_text(encoding="utf-8"))
    with phase_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_profile_and_shape = {
        (
            row["prompt_profile"],
            int(row["request_count"]),
            int(row["max_new_tokens"]),
        ): row
        for row in rows
    }
    shared_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == shared_profile
    }
    control_keys = {
        (request_count, max_new_tokens)
        for profile, request_count, max_new_tokens in by_profile_and_shape
        if profile == control_profile
    }
    comparison_keys = sorted(shared_keys & control_keys)
    if not comparison_keys:
        raise ValueError(
            "No matching shared/control profile rows found in prefix-cache phase comparison"
        )

    metric_fields = (
        "output_tokens_per_second_ratio",
        "p95_first_event_ms_ratio",
        "p95_latency_ms_ratio",
        "p95_stream_tpot_ms_ratio",
        "batch_wall_ms_ratio",
        "cold_prefix_cache_hit_rate_pct",
        "cache_prefix_cache_hit_rate_pct",
        "prefix_cache_hit_rate_pct_delta",
        "cold_gpu_kv_cache_usage_pct",
        "cache_gpu_kv_cache_usage_pct",
        "gpu_kv_cache_usage_pct_delta",
    )
    phase_labels = ("cold_first", "cache_first")
    comparison_rows = []
    for request_count, max_new_tokens in comparison_keys:
        shared = by_profile_and_shape[(shared_profile, request_count, max_new_tokens)]
        control = by_profile_and_shape[(control_profile, request_count, max_new_tokens)]
        shared_prompt_tokens = _float_field(shared, "prompt_tokens_mean")
        control_prompt_tokens = _float_field(control, "prompt_tokens_mean")
        row: dict[str, Any] = {
            "request_count": request_count,
            "max_new_tokens": max_new_tokens,
            "shared_profile": shared_profile,
            "control_profile": control_profile,
            "shared_scenario_id": shared["scenario_id"],
            "control_scenario_id": control["scenario_id"],
            "shared_prompt_tokens_mean": shared_prompt_tokens,
            "control_prompt_tokens_mean": control_prompt_tokens,
            "shared_to_control_prompt_tokens_mean_ratio": _ratio(
                shared_prompt_tokens,
                control_prompt_tokens,
            ),
        }
        for phase_label in phase_labels:
            for metric_field in metric_fields:
                source_field = f"{phase_label}_{metric_field}"
                shared_value = _float_field(shared, source_field)
                control_value = _float_field(control, source_field)
                row[f"shared_{source_field}"] = shared_value
                row[f"control_{source_field}"] = control_value
                row[f"shared_minus_control_{source_field}"] = _delta(
                    shared_value,
                    control_value,
                )
        comparison_rows.append(row)

    summary_fields = [
        field
        for field in comparison_rows[0]
        if field.startswith("shared_minus_control_")
    ]
    summary = {
        f"mean_{field}": _mean_present(row[field] for row in comparison_rows)
        for field in summary_fields
    }
    summary["mean_shared_to_control_prompt_tokens_mean_ratio"] = _mean_present(
        row["shared_to_control_prompt_tokens_mean_ratio"]
        for row in comparison_rows
    )

    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-profile-control",
        "phase_order_compare_dir": str(phase_order_compare_dir),
        "phase_order_compare_json": str(phase_json),
        "phase_order_compare_csv": str(phase_csv),
        "shared_profile": shared_profile,
        "control_profile": control_profile,
        "row_count": len(comparison_rows),
        "phase_order_scenario_count": phase_payload["scenario_count"],
        "summary": summary,
        "rows": comparison_rows,
    }


def _aggregate_vllm_prefix_cache_profile_control_trials(
    profile_control_dirs: list[Path],
) -> dict[str, Any]:
    if not profile_control_dirs:
        raise ValueError("profile_control_dirs must not be empty")

    metric_sources = (
        (
            "cold_first",
            "throughput_ratio",
            "mean_shared_minus_control_cold_first_output_tokens_per_second_ratio",
            "higher_is_better",
        ),
        (
            "cold_first",
            "first_event_ratio",
            "mean_shared_minus_control_cold_first_p95_first_event_ms_ratio",
            "lower_is_better",
        ),
        (
            "cold_first",
            "latency_ratio",
            "mean_shared_minus_control_cold_first_p95_latency_ms_ratio",
            "lower_is_better",
        ),
        (
            "cold_first",
            "tpot_ratio",
            "mean_shared_minus_control_cold_first_p95_stream_tpot_ms_ratio",
            "lower_is_better",
        ),
        (
            "cache_first",
            "throughput_ratio",
            "mean_shared_minus_control_cache_first_output_tokens_per_second_ratio",
            "higher_is_better",
        ),
        (
            "cache_first",
            "first_event_ratio",
            "mean_shared_minus_control_cache_first_p95_first_event_ms_ratio",
            "lower_is_better",
        ),
        (
            "cache_first",
            "latency_ratio",
            "mean_shared_minus_control_cache_first_p95_latency_ms_ratio",
            "lower_is_better",
        ),
        (
            "cache_first",
            "tpot_ratio",
            "mean_shared_minus_control_cache_first_p95_stream_tpot_ms_ratio",
            "lower_is_better",
        ),
    )
    trials = []
    for index, profile_control_dir in enumerate(profile_control_dirs, start=1):
        path = profile_control_dir / "prefix-cache-profile-control.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload["summary"]
        trial: dict[str, Any] = {
            "trial_index": index,
            "trial_label": profile_control_dir.name,
            "profile_control_json": str(path),
            "row_count": payload["row_count"],
            "shared_profile": payload["shared_profile"],
            "control_profile": payload["control_profile"],
            "mean_shared_to_control_prompt_tokens_mean_ratio": summary[
                "mean_shared_to_control_prompt_tokens_mean_ratio"
            ],
        }
        for phase_order, metric, source_field, _direction in metric_sources:
            trial[f"{phase_order}_{metric}"] = summary[source_field]
        trials.append(trial)

    summary_rows = []
    for phase_order, metric, _source_field, direction in metric_sources:
        field = f"{phase_order}_{metric}"
        stats = _metric_distribution(trials, field)
        values = [
            trial[field]
            for trial in trials
            if trial.get(field) is not None
        ]
        bootstrap_interval = _bootstrap_mean_interval(values)
        summary_rows.append(
            {
                "phase_order": phase_order,
                "metric": metric,
                "direction": direction,
                "trial_count": len(trials),
                "shared_minus_control_mean": stats[f"{field}_mean"],
                "shared_minus_control_min": stats[f"{field}_min"],
                "shared_minus_control_max": stats[f"{field}_max"],
                "shared_minus_control_cv": stats[f"{field}_cv"],
                "shared_minus_control_bootstrap_mean_p05": bootstrap_interval[
                    "mean_p05"
                ],
                "shared_minus_control_bootstrap_mean_p50": bootstrap_interval[
                    "mean_p50"
                ],
                "shared_minus_control_bootstrap_mean_p95": bootstrap_interval[
                    "mean_p95"
                ],
            }
        )

    prompt_ratio_stats = _metric_distribution(
        trials,
        "mean_shared_to_control_prompt_tokens_mean_ratio",
    )
    return {
        "schema_version": 1,
        "mode": "vllm-prefix-cache-profile-multitrial",
        "trial_count": len(trials),
        "profile_control_dirs": [str(path) for path in profile_control_dirs],
        "shared_profile": trials[0]["shared_profile"],
        "control_profile": trials[0]["control_profile"],
        "prompt_token_ratio_mean": prompt_ratio_stats[
            "mean_shared_to_control_prompt_tokens_mean_ratio_mean"
        ],
        "prompt_token_ratio_min": prompt_ratio_stats[
            "mean_shared_to_control_prompt_tokens_mean_ratio_min"
        ],
        "prompt_token_ratio_max": prompt_ratio_stats[
            "mean_shared_to_control_prompt_tokens_mean_ratio_max"
        ],
        "trials": trials,
        "summary": summary_rows,
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


def _mean_present(values: Any) -> float:
    present = [float(value) for value in values if value is not None]
    if not present:
        raise ValueError("Cannot compute mean for empty values")
    return sum(present) / len(present)


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
        writer = csv.DictWriter(
            handle,
            fieldnames=[*fieldnames, *extra_fields],
            lineterminator="\n",
        )
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
